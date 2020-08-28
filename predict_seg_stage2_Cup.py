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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

size = (512, 512)
Cup_dir = 'data/challenge/val/seg/stage2/Cup'
Disc_dir = 'data/challenge/val/seg/stage2/Disc'
Disc_Cup_dir = 'data/challenge/val/seg/stage2/Disc_Cup_respectively'
test_name_list = os.listdir('data/challenge/val/img')

os.makedirs(os.path.join(Cup_dir, 'predictions'), exist_ok=True)
os.makedirs(os.path.join(Disc_Cup_dir, 'predictions'), exist_ok=True)

model_paths = ['/home/zyw/refuge2/trained_models/seg/stage2/Cup/stage2_Cup_unet_resnet101_bs8_nbo1_size512_epoch150_fold0/best_dice.pth',
               '/home/zyw/refuge2/trained_models/seg/stage2/Cup/stage2_Cup_unet_resnet101_bs8_nbo1_size512_epoch150_fold1/best_dice.pth',
               '/home/zyw/refuge2/trained_models/seg/stage2/Cup/stage2_Cup_unet_resnet101_bs8_nbo1_size512_epoch150_fold2/best_dice.pth',
               '/home/zyw/refuge2/trained_models/seg/stage2/Cup/stage2_Cup_unet_resnet101_bs8_nbo1_size512_epoch150_fold3/best_dice.pth',
               '/home/zyw/refuge2/trained_models/seg/stage2/Cup/stage2_Cup_unet_resnet101_bs8_nbo1_size512_epoch150_fold4/best_dice.pth']

predictions_list = []
for name in test_name_list:
    img = cv2.imread(os.path.join('data/challenge/val/img', name))
    predictions_list.append(np.zeros(img.shape[:2]))

with torch.no_grad():
    for model_path in model_paths:
        if 'resnet34' in model_path:
            net = smp.Unet('resnet34', in_channels=3, classes=1, activation=None, encoder_weights=None).cuda()
        if 'resnet101' in model_path:
            net = smp.Unet('resnet101', in_channels=3, classes=1, activation=None, encoder_weights=None).cuda()
        if 'deeplab_resnet34' in model_path:
            net = smp.DeepLabV3Plus('resnet34', in_channels=3, classes=1, activation=None, encoder_weights=None).cuda()
        net.load_state_dict(torch.load(model_path))
        net.eval()
        for i, name in enumerate(test_name_list):
            print(i, name)
            img = io.imread(os.path.join('data/challenge/val/img', name)) / 255
            Disc_pred = cv2.imread(
                os.path.join('data/challenge/val/seg/stage2/Disc/predictions', name.rstrip('.jpg') + '.png'), 0)
            Disc_pred[Disc_pred == 255] = 0
            Disc_pred[Disc_pred == 128] = 1
            x1_new, x2_new, y1_new, y2_new = crop_mask_expand(Disc_pred, expand_Percentage=0.1)
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
                    img_crop_tta = exposure.adjust_gamma(img, 1.2)
                if t == 3:
                    img_crop_tta = exposure.adjust_gamma(img, 0.8)
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
    Disc_prediction = cv2.imread(os.path.join('data/challenge/val/seg/stage2/Disc/predictions', name.rstrip('.jpg') + '.png'), 0)
    Disc_mask = np.copy(Disc_prediction)
    Disc_mask[Disc_mask == 255] = 0
    Disc_mask[Disc_mask == 128] = 1
    predictions_list[i] *= Disc_mask
    predictions_list[i] = keep_large_area(predictions_list[i], 1)
    Cup_prediction = np.copy(predictions_list[i])
    Cup_prediction[Cup_prediction == 0] = 255
    Cup_prediction[Cup_prediction == 1] = 0
    Disc_Cup_prediction = np.copy(Disc_prediction)
    Disc_Cup_prediction[Cup_prediction == 0] = 0
    cv2.imwrite(os.path.join(os.path.join(Cup_dir, 'predictions', name.rstrip('.jpg') + '.png')), Cup_prediction)
    cv2.imwrite(os.path.join(os.path.join(Disc_Cup_dir, 'predictions', name.rstrip('.jpg') + '.png')), Disc_Cup_prediction)