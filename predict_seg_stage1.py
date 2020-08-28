import os
import numpy as np
import torch
import math
import segmentation_models_pytorch as smp
import pickle
import cv2
import argparse
from utils import remove_small_areas, keep_large_area
from skimage import exposure, io
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

new_dir = 'data/challenge/val/seg/stage1'
test_name_list = os.listdir('data/challenge/val/img')
input_size = [(384, 384), (512, 512), (640, 640)]

os.makedirs(new_dir, exist_ok=True)
os.makedirs(os.path.join(new_dir, 'predictions'), exist_ok=True)
os.makedirs(os.path.join(new_dir, 'uncertainty'), exist_ok=True)

model_paths = ['/home/zyw/refuge2/trained_models/seg/stage1/stage1_unet_resnet101_bs8_nbo1_epoch150_fold0/best_dice.pth',
               '/home/zyw/refuge2/trained_models/seg/stage1/stage1_unet_resnet101_bs8_nbo1_epoch150_fold1/best_dice.pth',
               '/home/zyw/refuge2/trained_models/seg/stage1/stage1_unet_resnet101_bs8_nbo1_epoch150_fold2/best_dice.pth',
               '/home/zyw/refuge2/trained_models/seg/stage1/stage1_unet_resnet101_bs8_nbo1_epoch150_fold3/best_dice.pth',
               '/home/zyw/refuge2/trained_models/seg/stage1/stage1_unet_resnet101_bs8_nbo1_epoch150_fold4/best_dice.pth']

predictions_list = []
uncertainty_list = []
repeat_list = [0] * len(test_name_list)
for name in test_name_list:
    img = cv2.imread(os.path.join('data/challenge/val/img', name), 0)
    predictions_list.append(np.zeros(img.shape))
    uncertainty_list.append(np.zeros(img.shape))

with torch.no_grad():
    for model_path in model_paths:
        if 'resnet34' in model_path:
            net = smp.Unet('resnet34', in_channels=3, classes=1, activation=None, encoder_weights=None).cuda()
        if 'resnet101' in model_path:
            net = smp.Unet('resnet101', in_channels=3, classes=1, activation=None, encoder_weights=None).cuda()
        net.load_state_dict(torch.load(model_path))
        net.eval()
        for i, name in enumerate(test_name_list):
            print(i, name)
            img = io.imread(os.path.join('data/challenge/val/img', name)) / 255
            repeat = 0
            pred_logits = np.zeros((img.shape[0], img.shape[1]))
            for s in input_size:
                img_resized = cv2.resize(img, (s[1], s[0]))
                for t in range(4):
                    if t == 0:
                        img_resized_tta = np.flip(img_resized, axis=0)
                    if t == 1:
                        img_resized_tta = exposure.adjust_gamma(img_resized, 1.2)
                    if t == 2:
                        img_resized_tta = exposure.adjust_gamma(img_resized, 0.8)
                    if t == 3:
                        img_resized_tta = img_resized
                    img_resized_tta = np.ascontiguousarray(img_resized_tta)
                    data_one_tensor = torch.from_numpy(img_resized_tta).permute(2, 0, 1).unsqueeze(0).float().cuda()
                    predict_one_tensor = net(data_one_tensor)
                    predict_one_tensor = torch.sigmoid(predict_one_tensor)
                    predict_one_array = predict_one_tensor.cpu().squeeze(0).squeeze(0).detach().numpy()
                    if t == 0:
                        predict_one_array = np.flip(predict_one_array, axis=0)
                    if predict_one_array.max() > 0.5:
                        repeat += 1
                        predict_one_array = cv2.resize(predict_one_array, (img.shape[1], img.shape[0]))
                        pred_logits += predict_one_array
            if repeat > 0:
                pred_logits /= repeat
                predictions_list[i] += pred_logits
                uncertainty_list[i] += pred_logits
                repeat_list[i] += 1

for i, name in enumerate(test_name_list):
    if repeat_list[i] > 0:
        predictions_list[i] /= repeat_list[i]
        uncertainty_list[i] /= repeat_list[i]
    _, predictions_list[i] = cv2.threshold(predictions_list[i], 0.5, 1, 0)
    predictions_list[i] = keep_large_area(predictions_list[i], 1)
    if predictions_list[i].sum() == 0:
        uncertainty = -uncertainty_list[i] * np.log2(uncertainty_list[i]) - (1 - uncertainty_list[i]) * np.log2(1 - uncertainty_list[i])
        cv2.imwrite(os.path.join(os.path.join(new_dir, 'uncertainty', name)), uncertainty * 255)
    predictions_list[i][predictions_list[i] == 0] = 255
    predictions_list[i][predictions_list[i] == 1] = 128
    cv2.imwrite(os.path.join(os.path.join(new_dir, 'predictions', name.rstrip('.jpg') + '.png')), predictions_list[i])