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

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=-1, help='fold of cross validation')
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
parser.add_argument('--net', type=str, default='Unet', help='net')
parser.add_argument('--model_path', type=str, help='trained model path')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

file = open(os.path.join('data/train_val_split_200803.pkl'), 'rb')
pkl_data = pickle.load(file)
test_name_list = pkl_data[args.fold][1]

input_size = [(384, 384), (512, 512), (640, 640)]

new_dir = args.model_path.rstrip('.pth')
os.makedirs(new_dir, exist_ok=True)
os.makedirs(os.path.join(new_dir, 'predictions'), exist_ok=True)
os.makedirs(os.path.join(new_dir, 'uncertainty'), exist_ok=True)
if os.path.exists(os.path.join(new_dir, 'evaluation.txt')):
    os.remove(os.path.join(new_dir, 'evaluation.txt'))
file = open(os.path.join(new_dir, 'evaluation.txt'), 'w')
dice_all = []

if args.net.lower() == 'unet_resnet34':
    net = smp.Unet('resnet34', in_channels=3, classes=1, activation=None, encoder_weights=None).cuda()
if args.net.lower() == 'unet_resnet101':
    net = smp.Unet('resnet101', in_channels=3, classes=1, activation=None, encoder_weights=None).cuda()

def compute_dice(pred, label):
    intersection = pred * label
    dice_sco = (2 * intersection.sum()) / (pred.sum() + label.sum())
    return dice_sco

net.load_state_dict(torch.load(args.model_path))
net.eval()

with torch.no_grad():
    for i, name in enumerate(test_name_list):
        img = io.imread(os.path.join('data/img', name)) / 255
        label = cv2.imread(os.path.join('data/mask', name.rstrip('.jpg') + '.png'), 0)
        label[label > 0.5] = 1
        pred_logits = np.zeros(label.shape)
        repeat = 0
        for size in input_size:
            img_resized = cv2.resize(img, (size[1], size[0]))
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
        _, predict_array = cv2.threshold(pred_logits, 0.5, 1, 0)
        predict_array = keep_large_area(predict_array, 1)
        dice = compute_dice(pred=predict_array, label=label)
        if predict_array.sum() == 0:
            uncertainty = -pred_logits * np.log2(pred_logits) - (1 - pred_logits) * np.log2(1 - pred_logits)
            cv2.imwrite(os.path.join(os.path.join(new_dir, 'uncertainty', name.rstrip('.jpg') + '.png')), uncertainty * 255)
        print(i, name, dice)
        dice_all.append(dice)
        file.write('{}: {}\n'.format(name, dice))
        cv2.imwrite(os.path.join(os.path.join(new_dir, 'predictions', name.rstrip('.jpg') + '.png')), predict_array * 255)
    file.write('mean: {}'.format(np.mean(dice_all)))
    file.close()