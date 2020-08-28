from torch.utils.data import Dataset, DataLoader
from albumentations import (ShiftScaleRotate, Compose, PadIfNeeded, RandomCrop, HorizontalFlip, OneOf, ElasticTransform,
                             OpticalDistortion, RandomGamma, Resize, GaussNoise, VerticalFlip, RandomBrightnessContrast)
import cv2
import os
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from skimage import io
from utils import remove_small_areas, keep_large_area, fit_Ellipse, crop_mask_expand, roi_extend

class Dataset_train(Dataset):
    def __init__(self, data_root='data', size=(384, 384), fold=0):
        self.root = data_root
        file = open(os.path.join(data_root, 'train_val_split_200803.pkl'), 'rb')
        pkl_data = pickle.load(file)
        self.size = size
        if fold == -1:
            self.train_name_list = pkl_data[0][0]
            self.train_name_list.append(pkl_data[0][1])
        else:
            self.train_name_list = pkl_data[fold][0]
        self.len = len(self.train_name_list)
        self.Fovea_location = pd.read_csv(os.path.join(data_root, 'Fovea_location.csv'))
        self.transforms = Compose([Resize(size[0], size[1]),
                                   ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=1,
                                                    border_mode=cv2.BORDER_CONSTANT, value=0),
                                   VerticalFlip(p=0.5),
                                   RandomGamma(gamma_limit=(80, 120), p=0.5),
                                   RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                                   GaussNoise(var_limit=(10, 100), mean=0, p=0.5)
                                   ])
        self.stage1_mask_transforms = Compose([ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=1,
                border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_NEAREST, value=0),
                OneOf([ElasticTransform(p=1, alpha=50, sigma=30, alpha_affine=30,
                                        border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_NEAREST, value=0),
                OpticalDistortion(p=1, distort_limit=0.5, shift_limit=0.1,
                                        border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_NEAREST, value=0)], p=0.5)])

    def __getitem__(self, idx):
        name = self.train_name_list[idx]
        if random.random() < 0.3:
            img = io.imread(os.path.join(self.root, 'img_match_challenge_val', name))
        else:
            img = io.imread(os.path.join(self.root, 'img', name))
        Fovea_x = float(self.Fovea_location.loc[self.Fovea_location.ImgName == name, 'Fovea_X'].values[0])
        Fovea_y = float(self.Fovea_location.loc[self.Fovea_location.ImgName == name, 'Fovea_Y'].values[0])
        Fovea_x = int(round(Fovea_x, 0))
        Fovea_y = int(round(Fovea_y, 0))
        stage1_mask = np.zeros(img.shape[:2])
        Fovea_mask = np.zeros(img.shape[:2])
        cv2.circle(stage1_mask, (Fovea_x, Fovea_y), radius=100, color=1, thickness=-1)
        cv2.circle(Fovea_mask, (Fovea_x, Fovea_y), radius=20, color=1, thickness=-1)
        x1_new, x2_new, y1_new, y2_new = crop_mask_expand(stage1_mask, expand_Percentage=0.5)
        stage1_mask_roi = stage1_mask[x1_new: x2_new, y1_new: y2_new]
        stage1_mask_roi = self.stage1_mask_transforms(image=stage1_mask_roi)['image']
        stage1_mask[x1_new: x2_new, y1_new: y2_new] = stage1_mask_roi
        x1_new, x2_new, y1_new, y2_new = crop_mask_expand(stage1_mask, expand_Percentage=0.1)
        x1_roi, x2_roi, y1_roi, y2_roi = roi_extend(img.shape[:2], self.size, x1_new, x2_new, y1_new, y2_new)
        img = img[x1_roi: x2_roi, y1_roi: y2_roi, :]
        Fovea_mask = Fovea_mask[x1_roi: x2_roi, y1_roi: y2_roi]
        stage1_mask = stage1_mask[x1_roi: x2_roi, y1_roi: y2_roi]
        Fovea_mask = Fovea_mask[:, :, np.newaxis]
        stage1_mask = stage1_mask[:, :, np.newaxis]
        Fovea_stage1_mask = np.concatenate((Fovea_mask, stage1_mask), axis=2)
        augmented = self.transforms(image=img, mask=Fovea_stage1_mask)
        img = augmented['image']
        Fovea_stage1_mask = augmented['mask']
        Fovea_mask = Fovea_stage1_mask[:, :, 0]
        stage1_mask = Fovea_stage1_mask[:, :, 1]
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255
        Fovea_mask = torch.from_numpy(Fovea_mask)
        stage1_mask = torch.from_numpy(stage1_mask)
        return img, Fovea_mask, stage1_mask

    def __len__(self):
        return self.len

class Dataset_val(Dataset):
    def __init__(self, data_root='data', size=(384, 384), fold=0):
        self.root = data_root
        file = open(os.path.join(data_root, 'train_val_split_200803.pkl'), 'rb')
        pkl_data = pickle.load(file)
        self.size = size
        self.val_name_list = pkl_data[fold][1]
        self.len = len(self.val_name_list)
        self.Fovea_location = pd.read_csv(os.path.join(data_root, 'Fovea_location.csv'))
        self.transforms = Compose([Resize(size[0], size[1]),
                                   ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5,
                                                    border_mode=cv2.BORDER_CONSTANT, value=0)
                                   ])
        self.stage1_mask_transforms = Compose([ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=1,
                                                                border_mode=cv2.BORDER_CONSTANT,
                                                                interpolation=cv2.INTER_NEAREST, value=0),
                                               OneOf([ElasticTransform(p=1, alpha=50, sigma=30, alpha_affine=30,
                                                                       border_mode=cv2.BORDER_CONSTANT,
                                                                       interpolation=cv2.INTER_NEAREST, value=0),
                                                      OpticalDistortion(p=1, distort_limit=0.5, shift_limit=0.1,
                                                                        border_mode=cv2.BORDER_CONSTANT,
                                                                        interpolation=cv2.INTER_NEAREST, value=0)],
                                                     p=0.5)])

    def __getitem__(self, idx):
        name = self.val_name_list[idx]
        if random.randint(0, 1) == 1:
            img = io.imread(os.path.join(self.root, 'img_match_challenge_val', name))
        else:
            img = io.imread(os.path.join(self.root, 'img', name))
        Fovea_x = float(self.Fovea_location.loc[self.Fovea_location.ImgName == name, 'Fovea_X'].values[0])
        Fovea_y = float(self.Fovea_location.loc[self.Fovea_location.ImgName == name, 'Fovea_Y'].values[0])
        Fovea_x = int(round(Fovea_x, 0))
        Fovea_y = int(round(Fovea_y, 0))
        stage1_mask = np.zeros(img.shape[:2])
        Fovea_mask = np.zeros(img.shape[:2])
        cv2.circle(stage1_mask, (Fovea_x, Fovea_y), radius=100, color=1, thickness=-1)
        cv2.circle(Fovea_mask, (Fovea_x, Fovea_y), radius=20, color=1, thickness=-1)
        x1_new, x2_new, y1_new, y2_new = crop_mask_expand(stage1_mask, expand_Percentage=0.5)
        stage1_mask_roi = stage1_mask[x1_new: x2_new, y1_new: y2_new]
        stage1_mask_roi = self.stage1_mask_transforms(image=stage1_mask_roi)['image']
        stage1_mask[x1_new: x2_new, y1_new: y2_new] = stage1_mask_roi
        x1_new, x2_new, y1_new, y2_new = crop_mask_expand(stage1_mask, expand_Percentage=0.1)
        x1_roi, x2_roi, y1_roi, y2_roi = roi_extend(img.shape[:2], self.size, x1_new, x2_new, y1_new, y2_new)
        img = img[x1_roi: x2_roi, y1_roi: y2_roi, :]
        Fovea_mask = Fovea_mask[x1_roi: x2_roi, y1_roi: y2_roi]
        stage1_mask = stage1_mask[x1_roi: x2_roi, y1_roi: y2_roi]
        Fovea_mask = Fovea_mask[:, :, np.newaxis]
        stage1_mask = stage1_mask[:, :, np.newaxis]
        Fovea_stage1_mask = np.concatenate((Fovea_mask, stage1_mask), axis=2)
        augmented = self.transforms(image=img, mask=Fovea_stage1_mask)
        img = augmented['image']
        Fovea_stage1_mask = augmented['mask']
        Fovea_mask = Fovea_stage1_mask[:, :, 0]
        stage1_mask = Fovea_stage1_mask[:, :, 1]
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255
        Fovea_mask = torch.from_numpy(Fovea_mask)
        stage1_mask = torch.from_numpy(stage1_mask)
        return img, Fovea_mask, stage1_mask

    def __len__(self):
        return self.len

if __name__ == '__main__':
    train_data = Dataset_val(data_root='data', size=(384, 384), fold=0)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    for i, data in enumerate(train_dataloader):
        input, labels, masks = data
        input = input.squeeze(0).permute(1, 2, 0).numpy()
        labels = labels.squeeze(0).numpy()
        masks = masks.squeeze(0).numpy()
        plt.subplot(131)
        plt.imshow(input)
        plt.subplot(132)
        plt.imshow(labels)
        plt.subplot(133)
        # plt.imshow(input + np.tile(labels[:, :, np.newaxis], (1, 1, 3)))
        plt.imshow(masks)
        plt.show()