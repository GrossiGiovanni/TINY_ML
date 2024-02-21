import os
import random
import torch.nn as nn
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from model import ENet
from config import cfg
from loading_data import loading_data
from utils import *
from timer import Timer
import pdb

exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH + '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()


def main():
    segmentation_type = "else"
    cfg_file = open('./config.py', "r")
    cfg_lines = cfg_file.readlines()

    with open(log_txt, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID) == 1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    net = []

    if cfg.TRAIN.STAGE == 'all':
        net = ENet(only_encode=False)
        if cfg.TRAIN.PRETRAINED_ENCODER != '':
            encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
            del encoder_weight['classifier.bias']
            del encoder_weight['classifier.weight']
            # pdb.set_trace()
            net.encoder.load_state_dict(encoder_weight)
    elif cfg.TRAIN.STAGE == 'encoder':
        net = ENet(only_encode=True).cuda()

    if len(cfg.TRAIN.GPU_ID) > 1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net = net.cuda()

    net.train()

    if segmentation_type == "binary":
        criterion = torch.nn.BCEWithLogitsLoss().cuda()  # Binary Classification
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=cfg.TRAIN.LABEL_WEIGHT.cuda()).cuda()

    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    _t = {'train time': Timer(), 'val time': Timer()}
    # validate(val_loader, net, criterion, optimizer, -1, restore_transform, segmentation_type)
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))
        _t['val time'].tic()
        validate(val_loader, net, criterion, optimizer, epoch, restore_transform, segmentation_type)
        _t['val time'].toc(average=False)
        print('val time of one epoch: {:.2f}s'.format(_t['val time'].diff))


def train(train_loader, net, criterion, optimizer, epoch):
    count = 1
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        


def validate(val_loader, net, criterion, optimizer, epoch,restore, seg_type):
    net.eval()
    criterion.cpu()
    iou_ = 0.0
    num_classes = 5
    iou_sum_classes = [0.0] * num_classes
    label_trues, label_preds = [], []
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda().long()
        outputs = net(inputs)
        # for binary classification
        if seg_type == "binary":
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            iou_ += calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()], [labels.data.cpu().numpy()], 2)
            pred_mask = outputs.squeeze_(1).cpu().numpy()
        else:
            for c in range(num_classes):
                # predmask
                pred_mask = (outputs.argmax(dim=1) == c).cpu().numpy()
                labels_mask = (labels == c).cpu().numpy()
                class_iou = calculate_mean_iu(pred_mask, labels_mask, 5)
                iou_sum_classes[c] += class_iou
            label_trues.append(labels.cpu().numpy())
            label_preds.append(outputs.argmax(dim=1).cpu().numpy())
            # if vi < 10:  # only do this for the first 10 examples
            # pred_mask = outputs.argmax(dim=1).cpu().numpy()
            # colored_mask = colorize_mask(pred_mask[0])  # colorize the first mask in the batch
            # colored_mask.show()  # display the mask
    print(scores(label_trues, label_preds, num_classes))

if __name__ == '__main__':
    main()
