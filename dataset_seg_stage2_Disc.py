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
from skimage import io

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
        if random.randint(0, 1) == 1:
            img = io.imread(os.path.join(self.root, 'img_match_challenge_val', name))
        else:
            img = io.imread(os.path.join(self.root, 'img', name))
        label = cv2.imread(os.path.join(self.root, 'mask', name.rstrip('.jpg') + '.png'), 0)  # [0, 100, 200]
        x1_new, x2_new, y1_new, y2_new = crop_mask_expand(label, expand_Percentage=0.1)
        x1_roi, x2_roi, y1_roi, y2_roi = roi_extend(img.shape[:2], self.size, x1_new, x2_new, y1_new, y2_new)
        img = img[x1_roi: x2_roi, y1_roi: y2_roi, :]
        label = label[x1_roi: x2_roi, y1_roi: y2_roi]
        augmented = self.transforms(image=img, mask=label)
        img, label = augmented['image'], augmented['mask']
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255
        label[label > 0] = 1
        label = torch.from_numpy(label)
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
        img = io.imread(os.path.join(self.root, 'img', name))
        label = cv2.imread(os.path.join(self.root, 'mask', name.rstrip('.jpg') + '.png'), 0)  # [0, 100, 200]
        stage1_pred = cv2.imread(os.path.join(self.root, 'stage1_pred', name.rstrip('.jpg') + '.png'), 0)
        _, stage1_pred = cv2.threshold(stage1_pred, 127, 1, 0)
        x1_new, x2_new, y1_new, y2_new = crop_mask_expand(stage1_pred, expand_Percentage=0.1)
        x1_roi, x2_roi, y1_roi, y2_roi = roi_extend(img.shape[:2], self.size, x1_new, x2_new, y1_new, y2_new)
        img = img[x1_roi: x2_roi, y1_roi: y2_roi, :]
        label = label[x1_roi: x2_roi, y1_roi: y2_roi]
        augmented = self.transforms(image=img, mask=label)
        img, label = augmented['image'], augmented['mask']
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255
        label[label > 0] = 1
        label = torch.from_numpy(label)
        return img, label

    def __len__(self):
        return self.len

if __name__ == '__main__':
    train_data = Dataset_val(data_root='data', size=(512, 512), fold=0)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    for i, data in enumerate(train_dataloader):
        input, labels = data
        img = input[:, :3, :, :]
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        labels = labels.squeeze(0).numpy()
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.subplot(122)
        plt.imshow(labels, cmap='gray')
        plt.show()