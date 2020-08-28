import os
from dataset_seg_stage2_Disc import Dataset_train, Dataset_val
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
from loss_function.pytorch_loss_function import dice_BCE_loss
from loss_function.DICE import dice1, DiceLoss
import segmentation_models_pytorch as smp
import math
import argparse
import shutil
import torch.nn as nn
from utils import adjust_lr

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=-1, help='fold of cross validation')
parser.add_argument('--gpu', type=str, default=0, help='which gpu is used')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--name', type=str, default='dp_10', help='net name')
parser.add_argument('--nbo', type=int, default=1, help='num_bs_opti')
parser.add_argument('--epoch', type=int, default=2000, help='all_epochs')
parser.add_argument('--net', type=str, default='Unet', help='net')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

input_size = (512, 512)
lr_max = 0.0001
data_path = 'data'
L2 = 0.0001
save_name = '{}_{}_bs{}_nbo{}_size{}_epoch{}_fold{}'.format(args.name, args.net, args.bs, args.nbo, input_size[0], args.epoch, args.fold)
os.makedirs(os.path.join('trained_models/seg/stage2/Disc', save_name), exist_ok=True)
train_writer = SummaryWriter(os.path.join('trained_models/seg/stage2/Disc', save_name, 'log/train'), flush_secs=2)
val_writer = SummaryWriter(os.path.join('trained_models/seg/stage2/Disc', save_name, 'log/val'), flush_secs=2)
print(save_name)

print('dataset loading')
train_data = Dataset_train(data_root=data_path, size=input_size, fold=args.fold)
train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, num_workers=32, pin_memory=True)
val_data = Dataset_val(data_root=data_path, size=input_size, fold=args.fold)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=False, num_workers=32, pin_memory=True)

print('model loading')
if args.net.lower() == 'unet_resnet34':
    net = smp.Unet('resnet34', in_channels=3, classes=1, activation=None)
if args.net.lower() == 'unet_resnet101':
    net = smp.Unet('resnet101', in_channels=3, classes=1, activation=None).cuda()
if args.net.lower() == 'deeplab_resnet34':
    net = smp.DeepLabV3Plus('resnet34', in_channels=3, classes=1, activation=None).cuda()

net.cuda()
train_data_len = train_data.len
val_data_len = val_data.len
print('train_lenth: %i  val_lenth: %i' % (train_data_len, val_data_len))

Dice_BCE_Loss = dice_BCE_loss(bce_weight=0.4, dice_weight=0.6)
optimizer = optim.Adam(net.parameters(), lr=lr_max, weight_decay=L2)
best_dice = 0

print('training')
for epoch in range(args.epoch):
    net.train()
    lr = adjust_lr(optimizer, lr_max, epoch, args.epoch)
    print('lr for this epoch:', lr)
    epoch_train_loss = []
    epoch_train_Disc_dice = []
    for i, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs, labels = inputs.float().cuda(), labels.float().cuda().unsqueeze(1)
        results = net(inputs)
        results = torch.sigmoid(results)
        dice_bce_loss = Dice_BCE_Loss(results, labels)
        train_loss_back = dice_bce_loss / args.nbo
        train_loss_back.backward()
        if (i + 1) % args.nbo == 0:
            optimizer.step()
        predictions = results.cpu().float()
        predictions[predictions <= 0.5] = 0
        predictions[predictions > 0.5] = 1
        Disc_dice = dice1(labels.cpu(), predictions).detach().numpy()
        epoch_train_loss.append(dice_bce_loss.item())
        epoch_train_Disc_dice.append(Disc_dice)
        print('[%d/%d, %5d/%d] train_loss: %.3f Disc_dice: %.3f ' % (epoch + 1, args.epoch, i + 1,
        math.ceil(train_data_len / args.bs), dice_bce_loss.item(), Disc_dice))

    with torch.no_grad():
        net.eval()
        epoch_val_loss = []
        epoch_val_Disc_dice = []
        for i, (inputs, labels) in enumerate(val_dataloader):
            inputs, labels = inputs.float().cuda(), labels.float().cuda().unsqueeze(1)
            results = net(inputs)
            results = torch.sigmoid(results)
            dice_bce_loss = Dice_BCE_Loss(results, labels)
            predictions = results.cpu().float()
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1
            Disc_dice = dice1(labels.cpu(), predictions).detach().numpy()
            epoch_val_loss.append(dice_bce_loss.item())
            epoch_val_Disc_dice.append(Disc_dice)
    epoch_train_loss = np.mean(epoch_train_loss)
    epoch_train_Disc_dice = np.mean(epoch_train_Disc_dice)
    epoch_val_loss = np.mean(epoch_val_loss)
    epoch_val_Disc_dice = np.mean(epoch_val_Disc_dice)
    print('[%d/%d] train_loss: %.3f Disc_dice: %.3f \n val_loss: %.3f Disc_dice: %.3f ' %
          (epoch + 1, args.epoch, epoch_train_loss, epoch_train_Disc_dice, epoch_val_loss, epoch_val_Disc_dice))
    train_writer.add_scalar('lr', lr, epoch)
    train_writer.add_scalar('loss', epoch_train_loss, epoch)
    train_writer.add_scalar('Disc_dice', epoch_train_Disc_dice, epoch)
    val_writer.add_scalar('loss', epoch_val_loss, epoch)
    val_writer.add_scalar('Disc_dice', epoch_val_Disc_dice, epoch)
    val_writer.add_scalar('best_dice', best_dice, epoch)
    if epoch + 1 == args.epoch:
        torch.save(net.state_dict(),
                   os.path.join('trained_models/seg/stage2/Disc', save_name, 'epoch' + str(epoch + 1) + '.pth'))
    if epoch_val_Disc_dice > best_dice:
        best_dice = epoch_val_Disc_dice
        torch.save(net.state_dict(),
                   os.path.join('trained_models/seg/stage2/Disc', save_name, 'best_dice.pth'))
train_writer.close()
val_writer.close()
print('saved_model_name:', save_name)