import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
import cv2
import pandas as pd
import argparse
from Network.Classification_Network import Resnet18, Resnet34, Se_Resnet50
from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

input_size = [(512, 512)]
new_dir = 'data/challenge/val/cate'
test_name_list = os.listdir('data/challenge/val/img')

os.makedirs(new_dir, exist_ok=True)
FileName = []
Glaucoma_Risk = []

net = Resnet34(classes=1).cuda()
model_paths = ['/home/zyw/refuge2/trained_models/Classification/cate_resnet34_bs48_nbo1_epoch300_fold0/best_AUC.pth',
               '/home/zyw/refuge2/trained_models/Classification/cate_resnet34_bs48_nbo1_epoch200_fold1/best_AUC.pth',
               '/home/zyw/refuge2/trained_models/Classification/cate_resnet34_bs48_nbo1_epoch200_fold2/best_AUC.pth']

with torch.no_grad():
    for i in range(400):
        name = 'V{}.jpg'.format(str(i + 1).zfill(4))
        img = io.imread(os.path.join('data/challenge/val/img', name))
        img_resized = cv2.resize(img, (512, 512)) / 255
        cate_logits = 0
        for model_path in model_paths:
            net.load_state_dict(torch.load(model_path))
            net.eval()
            data_one_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().cuda()
            cate_tensor = net(data_one_tensor)
            cate_tensor = torch.sigmoid(cate_tensor)
            cate_logits += float(cate_tensor.squeeze(1).cpu())
        cate_logits /= len(model_paths)
        FileName.append(name)
        Glaucoma_Risk.append(round(cate_logits, 4))
        print(name, round(cate_logits, 4))
save = pd.DataFrame({'FileName': FileName, 'Glaucoma Risk': Glaucoma_Risk})
save.to_csv(os.path.join(new_dir, 'classification_results.csv'), index=False)