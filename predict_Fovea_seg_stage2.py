import os
import numpy as np
import torch
import math
import pickle
import cv2
import argparse
from utils import remove_small_areas, keep_large_area, fit_Ellipse, crop_mask_expand, roi_extend
import segmentation_models_pytorch as smp
import shutil
import math
from skimage import exposure, io
import torch.nn as nn
from skimage.measure import *
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

stage2_dir = 'data/challenge/val/Fovea/stage2'
stage1_dir = 'data/challenge/val/Fovea/stage1'
test_name_list = os.listdir('data/challenge/val/img')

os.makedirs(os.path.join(stage2_dir, 'predictions'), exist_ok=True)

model_paths = ['/home/zyw/refuge2/trained_models/Fovea/stage2/stage2_seg_unet_resnet101_bs12_nbo1_size384_epoch200_fold-1/best_dice.pth']
               # '/home/zyw/refuge2/trained_models/Fovea/stage2/stage2_seg_deeplab_resnet34_bs46_nbo1_size384_epoch300_fold1/best_dice.pth']

ImageName = []
Fovea_X = []
Fovea_Y = []

predictions_list = []
repeat_list = [0] * len(test_name_list)
for name in test_name_list:
    img = cv2.imread(os.path.join('data/challenge/val/img', name), 0)
    predictions_list.append(np.zeros(img.shape))

with torch.no_grad():
    for model_path in model_paths:
        if 'unet_resnet34' in model_path:
            net = smp.Unet('resnet34', in_channels=3, classes=1, activation=None, encoder_weights=None).cuda()
        elif 'unet_resnet101' in model_path:
            net = smp.Unet('resnet101', in_channels=3, classes=1, activation=None, encoder_weights=None).cuda()
        elif 'deeplab_resnet34'in model_path:
            net = smp.DeepLabV3Plus('resnet34', in_channels=3, classes=1, activation=None, encoder_weights=None).cuda()
        if '512' in model_path:
            size = (512, 512)
        elif '384' in model_path:
            size = (384, 384)
        net.load_state_dict(torch.load(model_path))
        net.eval()
        for i, name in enumerate(test_name_list):
            print(i, name)
            img = io.imread(os.path.join('data/challenge/val/img', name)) / 255
            stage1_pred = cv2.imread(
                os.path.join(stage1_dir, 'predictions', name.rstrip('.jpg') + '.png'), 0)
            _, stage1_pred = cv2.threshold(stage1_pred, 127, 1, 0)
            x1_new, x2_new, y1_new, y2_new = crop_mask_expand(stage1_pred, expand_Percentage=0.1)
            predictions_stage2 = np.zeros(img.shape[:2])
            x1_roi, x2_roi, y1_roi, y2_roi = roi_extend(img.shape[:2], size, x1_new, x2_new, y1_new, y2_new)
            img = img[x1_roi: x2_roi, y1_roi: y2_roi, :]
            roi_shape = img.shape
            img = cv2.resize(img, (size[1], size[0]))
            pred_logits_roi = np.zeros((size[0], size[1]))
            for t in range(5):
                if t == 0:
                    img_crop_tta = np.flip(img, axis=0)
                if t == 1:
                    img_crop_tta = np.flip(img, axis=1)
                if t == 2:
                    img_crop_tta = exposure.adjust_gamma(img, 1.1)
                if t == 3:
                    img_crop_tta = exposure.adjust_gamma(img, 0.9)
                if t == 4:
                    img_crop_tta = img
                img_crop_tta = np.ascontiguousarray(img_crop_tta)
                img_crop_tensor = torch.from_numpy(img_crop_tta).permute(2, 0, 1).unsqueeze(0).float().cuda()
                predict_one_tensor = net(img_crop_tensor)
                predict_one_tensor = torch.sigmoid(predict_one_tensor)
                predict_one_array = predict_one_tensor.cpu().squeeze(0).squeeze(0).detach().numpy()
                if t == 0:
                    predict_one_array = np.flip(predict_one_array, axis=0)
                if t == 1:
                    predict_one_array = np.flip(predict_one_array, axis=1)
                pred_logits_roi += predict_one_array
            pred_logits_roi = cv2.resize(pred_logits_roi, (roi_shape[1], roi_shape[0]))
            predictions_stage2[x1_roi: x2_roi, y1_roi: y2_roi] = pred_logits_roi
            predictions_list[i] += predictions_stage2
for i, name in enumerate(test_name_list):
    predictions_list[i] = predictions_list[i] / (5 * len(model_paths))
    _, predictions_list[i] = cv2.threshold(predictions_list[i], 0.5, 1, 0)
    stage1_prediction = cv2.imread(os.path.join(stage1_dir, 'predictions', name.rstrip('.jpg') + '.png'), 0)
    _, stage1_prediction = cv2.threshold(stage1_prediction, 127, 1, 0)
    predictions_list[i] *= stage1_prediction
    predictions_list[i] = keep_large_area(predictions_list[i], 1)
    cv2.imwrite(os.path.join(os.path.join(stage2_dir, 'predictions', name.rstrip('.jpg') + '.png')), predictions_list[i] * 255)
    if predictions_list[i].max() > 0:
        prediction = predictions_list[i].astype(np.uint8)
    else:
        prediction = stage1_prediction.astype(np.uint8)
    connect_regions = label(prediction, connectivity=1, background=0)
    props = regionprops(connect_regions)
    Fovea_y, Fovea_x = props[0].centroid
    Fovea_x = round(Fovea_x, 2)
    Fovea_y = round(Fovea_y, 2)
    ImageName.append(name)
    Fovea_X.append(Fovea_x)
    Fovea_Y.append(Fovea_y)
save = pd.DataFrame({'ImageName': ImageName, 'Fovea_X': Fovea_X, 'Fovea_Y': Fovea_Y})
save.to_csv(os.path.join(stage2_dir, 'fovea_location_results.csv'), index=False)