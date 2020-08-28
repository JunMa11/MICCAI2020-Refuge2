from torch.utils.data import Dataset, DataLoader
from albumentations import (ShiftScaleRotate, Compose, RandomBrightnessContrast, RandomCrop, HorizontalFlip, OneOf, ElasticTransform,
                             OpticalDistortion, RandomGamma, Resize, GaussNoise, CoarseDropout, VerticalFlip)
import cv2
import os
import torch
import pickle
import matplotlib.pyplot as plt
from utils import remove_small_areas, keep_large_area, fit_Ellipse, crop_mask_expand, roi_extend
import numpy as np
import random
from skimage.transform import match_histograms

class Dataset_train(Dataset):
    def __init__(self, data_root='data', size=(512, 512), fold=0):
        self.root = data_root
        self.size = size
        file = open(os.path.join(data_root, 'train_val_split_200803.pkl'), 'rb')
        pkl_data = pickle.load(file)
        if fold == -1:
            self.train_name_list = pkl_data[0][0]
            self.train_name_list.append(pkl_data[0][1])
        else:
            self.train_name_list = pkl_data[fold][0]
        self.len = len(self.train_name_list)
        self.challenge_val_name_list = os.listdir(os.path.join(data_root, 'challenge/val/img'))
        self.transforms = Compose([Resize(size[0], size[1]),
                                   ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=180, p=0.7,
                                                    border_mode=cv2.BORDER_CONSTANT, value=0),
                                   HorizontalFlip(p=0.5),
                                   VerticalFlip(p=0.5),
                                   OneOf([ElasticTransform(p=1, alpha=50, sigma=30, alpha_affine=30,
                                                           border_mode=cv2.BORDER_CONSTANT, value=0),
                                          OpticalDistortion(p=1, distort_limit=0.5, shift_limit=0.1,
                                                            border_mode=cv2.BORDER_CONSTANT, value=0)], p=0.5),
                                   RandomGamma(gamma_limit=(80, 120), p=0.5),
                                   RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                                   GaussNoise(var_limit=(10, 100), mean=0, p=0.5)
                                   ])

    def __getitem__(self, idx):
        name = self.train_name_list[idx]
        img = cv2.imread(os.path.join(self.root, 'img', name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if random.randint(0, 1) == 1:
            challenge_val_name = random.sample(self.challenge_val_name_list, 1)[0]
            challenge_val_img = cv2.imread(os.path.join(self.root, 'challenge/val/img', challenge_val_name))
            challenge_val_img = cv2.cvtColor(challenge_val_img, cv2.COLOR_BGR2RGB)
            img = match_histograms(img, challenge_val_img, multichannel=True)
            img = img.astype(np.uint8)
        label = cv2.imread(os.path.join(self.root, 'mask', name.rstrip('.jpg') + '.png'), 0)  #[0, 100, 200]
        Ellipse_mask = np.copy(label)
        Ellipse_mask[Ellipse_mask > 0] = 1
        Ellipse_mask = fit_Ellipse(Ellipse_mask)
        x1_new, x2_new, y1_new, y2_new = crop_mask_expand(Ellipse_mask, expand_Percentage=0.1)
        Ellipse_mask_croped_roi = Ellipse_mask[x1_new: x2_new, y1_new: y2_new]
        make_hole_for_mask = Compose([ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=180, p=0.8,
                                                       border_mode=cv2.BORDER_CONSTANT, value=0),
                                      CoarseDropout(max_holes=4, max_height=int(Ellipse_mask_croped_roi.shape[0] / 2),
                                                    max_width=int(Ellipse_mask_croped_roi.shape[1] / 2),
                                                    min_holes=1, min_height=5, min_width=5, fill_value=0, p=0.8)
                                      ])
        Ellipse_mask_croped_roi = make_hole_for_mask(image=Ellipse_mask_croped_roi)['image']
        Ellipse_mask[x1_new: x2_new, y1_new: y2_new] = Ellipse_mask_croped_roi
        x1_roi, x2_roi, y1_roi, y2_roi = roi_extend(img.shape[:2], self.size, x1_new, x2_new, y1_new, y2_new)
        img = img[x1_roi: x2_roi, y1_roi: y2_roi, :]
        label = label[x1_roi: x2_roi, y1_roi: y2_roi]
        Ellipse_mask = Ellipse_mask[x1_roi: x2_roi, y1_roi: y2_roi]
        label_Ellipse = np.concatenate((label[:, :, np.newaxis], Ellipse_mask[:, :, np.newaxis]), axis=2)
        augmented = self.transforms(image=img, mask=label_Ellipse)
        img, label_Ellipse = augmented['image'], augmented['mask']
        label = label_Ellipse[:, :, 0]
        Ellipse_mask = label_Ellipse[:, :, 1]
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        label = torch.from_numpy(label).long() / 100
        Ellipse_mask = torch.from_numpy(Ellipse_mask).float().unsqueeze(0)
        img = img / 255
        img = torch.cat((img, Ellipse_mask), 0)
        return img, label

    def __len__(self):
        return self.len

class Dataset_val(Dataset):
    def __init__(self, data_root='data', size=(512, 512), fold=0):
        self.root = data_root
        self.size = size
        file = open(os.path.join(data_root, 'train_val_split_200803.pkl'), 'rb')
        pkl_data = pickle.load(file)
        self.val_name_list = pkl_data[fold][1]
        self.len = len(self.val_name_list)
        self.transforms = Compose([Resize(size[0], size[1])])

    def __getitem__(self, idx):
        name = self.val_name_list[idx]
        img = cv2.imread(os.path.join(self.root, 'img', name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.root, 'mask', name.rstrip('.jpg') + '.png'), 0)  #[0, 100, 200]
        stage1_pred = cv2.imread(os.path.join(self.root, 'stage1_pred', name), 0)
        _, stage1_pred = cv2.threshold(stage1_pred, 127, 1, 0)
        x1_new, x2_new, y1_new, y2_new = crop_mask_expand(stage1_pred, expand_Percentage=0.1)
        x1_roi, x2_roi, y1_roi, y2_roi = roi_extend(img.shape[:2], self.size, x1_new, x2_new, y1_new, y2_new)
        img = img[x1_roi: x2_roi, y1_roi: y2_roi, :]
        label = label[x1_roi: x2_roi, y1_roi: y2_roi]
        stage1_pred = stage1_pred[x1_roi: x2_roi, y1_roi: y2_roi]
        label_pred = np.concatenate((label[:, :, np.newaxis], stage1_pred[:, :, np.newaxis]), axis=2)
        augmented = self.transforms(image=img, mask=label_pred)
        img, label_pred = augmented['image'], augmented['mask']
        label = label_pred[:, :, 0]
        stage1_pred = label_pred[:, :, 1]
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255
        label = torch.from_numpy(label).long() / 100
        stage1_pred = torch.from_numpy(stage1_pred).float().unsqueeze(0)
        img = torch.cat((img, stage1_pred), 0)
        return img, label

    def __len__(self):
        return self.len

if __name__ == '__main__':
    train_data = Dataset_val(data_root='data', size=(512, 512), fold=0)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    for i, data in enumerate(train_dataloader):
        input, labels = data
        img = input[:, :3, :, :]
        Ellipse_mask = input[:, 3, :, :]
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        labels = labels.squeeze(0).numpy()
        Ellipse_mask = Ellipse_mask.squeeze(0).numpy()
        plt.subplot(131)
        plt.imshow(img, cmap='gray')
        plt.subplot(132)
        plt.imshow(labels, cmap='gray')
        plt.subplot(133)
        plt.imshow(Ellipse_mask, cmap='gray')
        plt.show()