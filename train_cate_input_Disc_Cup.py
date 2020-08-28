import os
from dataset_cate_input_Disc_Cup_mask import Dataset_train, Dataset_val
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
import math
from efficientnet_pytorch import EfficientNet
import argparse
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from utils import adjust_lr
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='fold of cross validation')
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--name', type=str, default='input_Disc_Cup', help='net name')
parser.add_argument('--nbo', type=int, default=1, help='num_bs_opti')
parser.add_argument('--epoch', type=int, default=200, help='all_epochs')
parser.add_argument('--net', type=str, default='efficientnet-b6', help='net')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

input_size = (512, 512)
lr_max = 0.0001
data_path = 'data'
L2 = 0.0002

save_name = '{}_{}_bs{}_nbo{}_size{}_epoch{}_fold{}'.format(args.name, args.net, args.bs, args.nbo, input_size[0], args.epoch, args.fold)
os.makedirs(os.path.join('trained_models/Classification', save_name), exist_ok=True)
train_writer = SummaryWriter(os.path.join('trained_models/Classification', save_name, 'log/train'), flush_secs=2)
val_writer = SummaryWriter(os.path.join('trained_models/Classification', save_name, 'log/val'), flush_secs=2)
print(save_name)

print('dataset loading')
train_data = Dataset_train(data_root=data_path, fold=args.fold)
train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, num_workers=16, pin_memory=True)
val_data = Dataset_val(data_root=data_path, fold=args.fold)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=False, num_workers=16, pin_memory=True)

if args.net.lower() == 'efficientnet-b6':
    net = EfficientNet.from_pretrained('efficientnet-b6', num_classes=1, in_channels=5).cuda()

train_data_len = train_data.len
val_data_len = val_data.len
print('train_lenth: %i  val_lenth: %i' % (train_data_len, val_data_len))

# criterion = nn.CrossEntropyLoss()
BCELoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9]).cuda())
optimizer = optim.Adam(net.parameters(), lr=lr_max, weight_decay=L2)
best_AUC = 0

print('training')
for epoch in range(args.epoch):
    net.train()
    lr = adjust_lr(optimizer, lr_max, epoch, args.epoch)
    print('lr for this epoch:', lr)
    epoch_train_loss = []
    epoch_train_label = []
    epoch_train_pred_scores = []
    optimizer.zero_grad()
    for i, (inputs, cate) in enumerate(train_dataloader):
        inputs, cate = inputs.float().cuda(), cate.float().cuda().unsqueeze(1)
        cate_logits = net(inputs)
        train_loss = BCELoss(cate_logits, cate)
        train_loss_back = train_loss / args.nbo
        train_loss_back.backward()
        if (i + 1) % args.nbo == 0:
            optimizer.step()
            optimizer.zero_grad()
        pred_scores = torch.sigmoid(cate_logits)
        for n in range(inputs.size(0)):
            epoch_train_label.append(int(cate[n][0].cpu()))
            epoch_train_pred_scores.append(float(pred_scores[n][0].cpu()))
        epoch_train_loss.append(train_loss.item())
        print('[%d/%d, %5d/%d] train_loss: %.3f' %
              (epoch + 1, args.epoch, i + 1, math.ceil(train_data_len / args.bs), train_loss.item()))

    with torch.no_grad():
        net.eval()
        epoch_val_loss = []
        epoch_val_label = []
        epoch_val_pred_scores = []
        for i, (inputs, cate) in enumerate(val_dataloader):
            inputs, cate = inputs.float().cuda(), cate.float().cuda().unsqueeze(1)
            cate_logits = net(inputs)
            val_loss = BCELoss(cate_logits, cate)
            pred_scores = torch.sigmoid(cate_logits)
            for n in range(inputs.size(0)):
                epoch_val_label.append(int(cate[n][0].cpu()))
                epoch_val_pred_scores.append(float(pred_scores[n][0].cpu()))
            epoch_val_loss.append(val_loss.item())
    epoch_train_loss = np.mean(epoch_train_loss)
    epoch_val_loss = np.mean(epoch_val_loss)
    train_AUC = roc_auc_score(np.array(epoch_train_label), np.array(epoch_train_pred_scores), average='weighted')
    val_AUC = roc_auc_score(np.array(epoch_val_label), np.array(epoch_val_pred_scores), average='weighted')
    print('[%d/%d] train_loss: %.3f train_ACU: %.3f val_loss: %.3f  val_AUC: %.3f' %
          (epoch + 1, args.epoch, epoch_train_loss, train_AUC, epoch_val_loss, val_AUC))
    train_writer.add_scalar('lr', lr, epoch)
    train_writer.add_scalar('loss', epoch_train_loss, epoch)
    train_writer.add_scalar('AUC', train_AUC, epoch)
    val_writer.add_scalar('loss', epoch_val_loss, epoch)
    val_writer.add_scalar('AUC', val_AUC, epoch)
    val_writer.add_scalar('best_AUC', best_AUC, epoch)
    if epoch + 1 == args.epoch:
        torch.save(net.state_dict(),
                   os.path.join('trained_models/Classification', save_name, 'epoch' + str(epoch + 1) + '.pth'))
    if val_AUC > best_AUC:
        best_AUC = val_AUC
        torch.save(net.state_dict(),
                   os.path.join('trained_models/Classification', save_name, 'best_AUC.pth'))
train_writer.close()
val_writer.close()
print('saved_model_name:', save_name)