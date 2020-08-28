from torch.utils.data import Dataset, DataLoader
from albumentations import (ShiftScaleRotate, Compose, CoarseDropout, RandomCrop, HorizontalFlip, OneOf, ElasticTransform,
                             OpticalDistortion, RandomGamma, Resize, GaussNoise, VerticalFlip, RandomBrightnessContrast)
import cv2
import os
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import random
from utils import remove_small_areas, keep_large_area, fit_Ellipse, crop_mask_expand, roi_extend

class Dataset_train(Dataset):
    def __init__(self, data_root='data', size=(512, 512), fold=0):
        self.root = data_root
        file = open(os.path.join(data_root, 'train_val_split_200803.pkl'), 'rb')
        pkl_data = pickle.load(file)
        if fold == -1:
            self.train_name_list = pkl_data[0][0]
            self.train_name_list.append(pkl_data[0][1])
        else:
            self.train_name_list = pkl_data[fold][0]
        self.len = len(self.train_name_list)
        self.transforms = Compose([Resize(size[0], size[0]),
                                   ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7,
                                                    border_mode=cv2.BORDER_CONSTANT, value=0),
                                   VerticalFlip(p=0.5),
                                   OneOf([ElasticTransform(p=1, alpha=50, sigma=30, alpha_affine=30,
                                                           border_mode=cv2.BORDER_CONSTANT, value=0),
                                          OpticalDistortion(p=1, distort_limit=0.5, shift_limit=0.1,
                                                            border_mode=cv2.BORDER_CONSTANT, value=0)], p=0.5),
                                   RandomGamma(gamma_limit=(80, 120), p=0.5),
                                   GaussNoise(var_limit=(10, 100), mean=0, p=0.5),
                                   RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                                   CoarseDropout(max_holes=2, max_height=256, max_width=256, min_holes=1, min_height=5,
                                                 min_width=5, fill_value=0, p=0.5)
                                   ])
        self.pseudo_mask_transformation = Compose([ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=180, p=0.7,
             border_mode=cv2.BORDER_CONSTANT, value=0, interpolation=cv2.INTER_NEAREST),
             OneOf([ElasticTransform(p=1, alpha=50, sigma=30, alpha_affine=30, border_mode=cv2.BORDER_CONSTANT,
                                     value=0, interpolation=cv2.INTER_NEAREST),
                    OpticalDistortion(p=1, distort_limit=0.5, shift_limit=0.1, border_mode=cv2.BORDER_CONSTANT,
                                      value=0, interpolation=cv2.INTER_NEAREST)], p=0.5)])

    def __getitem__(self, idx):
        name = self.train_name_list[idx]
        if random.randint(0, 1) == 1:
            img = io.imread(os.path.join(self.root, 'img_match_challenge_val', name))
        else:
            img = io.imread(os.path.join(self.root, 'img', name))
        Disc_Cup_mask = cv2.imread(os.path.join(self.root, 'mask', name.rstrip('.jpg') + '.png'), 0)  # [0, 100, 200]
        cate = int(name.strip('.jpg').split('_')[-1])
        x1_new, x2_new, y1_new, y2_new = crop_mask_expand(Disc_Cup_mask, expand_Percentage=0.2)
        Disc_Cup_mask_ROI = Disc_Cup_mask[x1_new: x2_new, y1_new: y2_new]
        Disc_Cup_mask_ROI = self.pseudo_mask_transformation(image=Disc_Cup_mask_ROI)['image']
        Disc_Cup_mask[x1_new: x2_new, y1_new: y2_new] = Disc_Cup_mask_ROI
        augmented = self.transforms(image=img, mask=Disc_Cup_mask)
        img, Disc_Cup_mask = augmented['image'], augmented['mask']
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255
        Disc_mask = (Disc_Cup_mask > 0).astype(np.uint8)
        Cup_mask = (Disc_Cup_mask == 200).astype(np.uint8)
        Disc_mask = torch.from_numpy(Disc_mask).unsqueeze(0).float()
        Cup_mask = torch.from_numpy(Cup_mask).unsqueeze(0).float()
        img = torch.cat((img, Disc_mask, Cup_mask), dim=0)
        cate = torch.tensor(cate)
        return img, cate

    def __len__(self):
        return self.len

class Dataset_val(Dataset):
    def __init__(self, data_root='data', size=(512, 512), fold=0):
        self.root = data_root
        file = open(os.path.join(data_root, 'train_val_split_200803.pkl'), 'rb')
        pkl_data = pickle.load(file)
        self.val_name_list = pkl_data[fold][1]
        self.len = len(self.val_name_list)
        self.transforms = Compose([Resize(size[0], size[1])])
        self.pseudo_mask_transformation = Compose(
            [ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=180, p=0.7,
                              border_mode=cv2.BORDER_CONSTANT, value=0, interpolation=cv2.INTER_NEAREST),
             OneOf([ElasticTransform(p=1, alpha=50, sigma=30, alpha_affine=30, border_mode=cv2.BORDER_CONSTANT,
                                     value=0, interpolation=cv2.INTER_NEAREST),
                    OpticalDistortion(p=1, distort_limit=0.5, shift_limit=0.1, border_mode=cv2.BORDER_CONSTANT,
                                      value=0, interpolation=cv2.INTER_NEAREST)], p=0.5)])

    def __getitem__(self, idx):
        name = self.val_name_list[idx]
        if random.randint(0, 1) == 1:
            img = io.imread(os.path.join(self.root, 'img_match_challenge_val', name))
        else:
            img = io.imread(os.path.join(self.root, 'img', name))
        Disc_Cup_mask = cv2.imread(os.path.join(self.root, 'mask', name.rstrip('.jpg') + '.png'), 0)  # [0, 100, 200]
        cate = int(name.strip('.jpg').split('_')[-1])
        x1_new, x2_new, y1_new, y2_new = crop_mask_expand(Disc_Cup_mask, expand_Percentage=0.2)
        Disc_Cup_mask_ROI = Disc_Cup_mask[x1_new: x2_new, y1_new: y2_new]
        Disc_Cup_mask_ROI = self.pseudo_mask_transformation(image=Disc_Cup_mask_ROI)['image']
        Disc_Cup_mask[x1_new: x2_new, y1_new: y2_new] = Disc_Cup_mask_ROI
        augmented = self.transforms(image=img, mask=Disc_Cup_mask)
        img, Disc_Cup_mask = augmented['image'], augmented['mask']
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255
        Disc_mask = (Disc_Cup_mask > 0).astype(np.uint8)
        Cup_mask = (Disc_Cup_mask == 200).astype(np.uint8)
        Disc_mask = torch.from_numpy(Disc_mask).unsqueeze(0).float()
        Cup_mask = torch.from_numpy(Cup_mask).unsqueeze(0).float()
        img = torch.cat((img, Disc_mask, Cup_mask), dim=0)
        cate = torch.tensor(cate)
        return img, cate

    def __len__(self):
        return self.len

if __name__ == '__main__':
    train_data = Dataset_val(data_root='data', size=(512, 512), fold=0)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    for i, (inputs, cate) in enumerate(train_dataloader):
        print(cate, cate.size())
        img = inputs[:, :3, :, :].squeeze(0).permute(1, 2, 0).numpy()
        Disc = inputs[:, 3, :, :].squeeze(0).numpy()
        Cup = inputs[:, 4, :, :].squeeze(0).numpy()
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(Disc)
        plt.subplot(133)
        plt.imshow(Cup)
        plt.show()