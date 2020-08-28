import os
from dataset_Fovea_seg_stage1 import Dataset_train, Dataset_val
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
from loss_function.pytorch_loss_function import dice_BCE_loss
from loss_function.DICE import dice1
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import segmentation_models_pytorch as smp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='fold of cross validation')
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
parser.add_argument('--bs', type=int, default=10, help='batch size')
parser.add_argument('--name', type=str, default='1', help='net name')
parser.add_argument('--nbo', type=int, default=2, help='num_bs_opti')
parser.add_argument('--epoch', type=int, default=300, help='all_epochs')
parser.add_argument('--net', type=str, default='unet', help='net')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

input_size = (512, 512)
lr_max = 0.0001
data_path = 'data'
L2 = 0.0001
save_name = '{}_{}_bs{}_nbo{}_epoch{}_fold{}'.format(args.name, args.net, args.bs, args.nbo, args.epoch, args.fold)
os.makedirs(os.path.join('trained_models/Fovea', save_name), exist_ok=True)
train_writer = SummaryWriter(os.path.join('trained_models/Fovea', save_name, 'log/train'), flush_secs=2)
val_writer = SummaryWriter(os.path.join('trained_models/Fovea', save_name, 'log/val'), flush_secs=2)
print(save_name)

print('dataset loading')
train_data = Dataset_train(data_root=data_path, size=input_size, fold=args.fold)
train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, num_workers=16, pin_memory=True)
val_data = Dataset_val(data_root=data_path, size=input_size, fold=args.fold)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=False, num_workers=16, pin_memory=True)

print('model loading')
if args.net.lower() == 'unet_resnet34':
    net = smp.Unet('resnet34', in_channels=3, classes=1, activation=None).cuda()
if args.net.lower() == 'unet_resnet101':
    net = smp.Unet('resnet101', in_channels=3, classes=1, activation=None).cuda()
if args.net.lower() == 'deeplab_resnet34':
    net = smp.DeepLabV3Plus('resnet34', in_channels=3, classes=1, activation=None).cuda()

train_data_len = train_data.len
val_data_len = val_data.len
print('train_lenth: %i  val_lenth: %i' % (train_data_len, val_data_len))

def adjust_lr(optimizer, lr_max, epoch, all_epochs):
    if epoch < 0.1 * all_epochs:
        lr = (0.99 * lr_max * epoch) / (0.1 * all_epochs) + 0.01 * lr_max
    elif epoch < 0.6 * all_epochs:
        lr = lr_max
    elif epoch < 0.9 * all_epochs:
        lr = lr_max * 0.1
    else:
        lr = lr_max * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

criterion = dice_BCE_loss(0.2, 0.8)
optimizer = optim.Adam(net.parameters(), lr=lr_max, weight_decay=L2)
best_dice = 0

print('training')
for epoch in range(args.epoch):
    net.train()
    lr = adjust_lr(optimizer, lr_max, epoch, args.epoch)
    print('lr for this epoch:', lr)
    epoch_train_loss = []
    epoch_train_dice = []
    optimizer.zero_grad()
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.float().cuda(), labels.float().cuda()
        labels = labels.unsqueeze(1)
        result = net(inputs)
        result = torch.sigmoid(result)
        train_loss = criterion(result, labels)
        train_loss_back = train_loss / args.nbo
        train_loss_back.backward()
        if (i + 1) % args.nbo == 0:
            optimizer.step()
            optimizer.zero_grad()
        pred_train = result.cpu().float()
        pred_train[pred_train <= 0.5] = 0
        pred_train[pred_train > 0.5] = 1
        pred_dice = dice1(labels.cpu(), pred_train).detach().numpy()
        epoch_train_dice.append(pred_dice)
        epoch_train_loss.append(train_loss.item())
        print('[%d/%d, %5d/%d] train_loss: %.3f train_dice: %.3f' % (epoch + 1, args.epoch, i + 1,
            math.ceil(train_data_len/args.bs), train_loss.item(), pred_dice))

    net.eval()
    epoch_val_loss = []
    epoch_val_dice = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dataloader):
            inputs, labels = inputs.float().cuda(), labels.float().cuda()
            labels = labels.unsqueeze(1)
            result = net(inputs)
            result = torch.sigmoid(result)
            val_loss = criterion(result, labels)
            pred_val = result.cpu().float()
            pred_val[pred_val <= 0.5] = 0
            pred_val[pred_val > 0.5] = 1
            pred_dice = dice1(labels.cpu(), pred_val).detach().numpy()
            epoch_val_dice.append(pred_dice)
            epoch_val_loss.append(val_loss.item())
    epoch_train_loss = np.mean(epoch_train_loss)
    epoch_train_dice = np.mean(epoch_train_dice)
    epoch_val_loss = np.mean(epoch_val_loss)
    epoch_val_dice = np.mean(epoch_val_dice)
    print('[%d/%d] train_loss: %.3f  val_loss: %.3f  val_dice: %.3f' % (epoch + 1, args.epoch, epoch_train_loss,
        epoch_val_loss, epoch_val_dice))
    train_writer.add_scalar('lr', lr, epoch)
    train_writer.add_scalar('loss', epoch_train_loss, epoch)
    train_writer.add_scalar('dice', epoch_train_dice, epoch)
    val_writer.add_scalar('loss', epoch_val_loss, epoch)
    val_writer.add_scalar('dice', epoch_val_dice, epoch)
    val_writer.add_scalar('best_dice', best_dice, epoch)
    if epoch + 1 == args.epoch:
        torch.save(net.state_dict(),
                   os.path.join('trained_models/Fovea', save_name, 'epoch' + str(epoch + 1) + '.pth'))
    if epoch_val_dice > best_dice:
        best_dice = epoch_val_dice
        torch.save(net.state_dict(),
                   os.path.join('trained_models/Fovea', save_name, 'best_dice.pth'))
train_writer.close()
val_writer.close()
print('saved_model_name:', save_name)

