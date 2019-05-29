from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from faster_rcnn.faster_rcnn import FasterRCNN
from dataset.roidb import combined_roidb, RoiDataset
from config.config import cfg
import time


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_args():
    parser = argparse.ArgumentParser(description='Faster RCNN')
    parser.add_argument('--dataset', dest='dataset', default='voc0712trainval', type=str)
    parser.add_argument('--backbone', dest='backbone', default='resnet101', type=str)
    parser.add_argument('--use_gpu', dest='use_gpu', default=True, type=bool)
    parser.add_argument('--batch_size', dest='batch_size', default=1, type=int)
    parser.add_argument('--epochs', dest='epochs', default=10, type=int)
    parser.add_argument('--lr', dest='lr', default=0.001, type=float)
    parser.add_argument('--decay_lrs', dest='decay_lrs', default=[7, 9])
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.0005, type=float)
    parser.add_argument('--bais_decay', dest='bais_decay', default=False, type=bool)
    parser.add_argument('--gamma', dest='gamma', default=0.1, type=float)
    parser.add_argument('--use_tfboard', dest='use_tfboard', default=True, type=bool)
    parser.add_argument('--display_interval', dest='display_interval', default=100, type=int)
    parser.add_argument('--output_dir', dest='output_dir', default='output', type=str)
    parser.add_argument('--save_interval', dest='save_interval', default=1, type=int)

    args = parser.parse_args()
    return args


def train():

    args = parse_args()
    lr = args.lr
    decay_lrs = args.decay_lrs
    momentum = args.momentum
    weight_decay = args.weight_decay
    bais_decay = args.bais_decay
    gamma = args.gamma

    if args.use_tfboard:
        writer = SummaryWriter()

    # load data
    print('load data')
    if args.dataset == 'voc07trainval':
        dataset_name = 'voc_2007_trainval'
    elif args.dataset == 'voc12trainval':
        dataset_name = 'voc_2012_trainval'
    elif args.dataset == 'voc0712trainval':
        dataset_name = 'voc_2007_trainval+voc_2012_trainval'
    else:
        raise NotImplementedError

    imdb, roidb = combined_roidb(dataset_name)
    train_dataset = RoiDataset(roidb)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    iter_per_epoch = int(len(train_dataset))

    # prepare model
    print('load model')
    model = FasterRCNN(backbone=args.backbone)
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                        'weight_decay': bais_decay and weight_decay or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

    if args.use_gpu:
        model = model.cuda()

    model.train()

    # define optimizer
    optimizer = SGD(params, momentum)

    # training
    print('start training...')
    for epoch in range(args.epochs):
        start_time = time.time()
        train_data_iter = iter(train_dataloader)
        temp_loss = 0
        rpn_tp, rpn_tn, rpn_fg, rpn_bg = 0, 0, 0, 0
        faster_rcnn_tp, faster_rcnn_tn, faster_rcnn_fg, faster_rcnn_bg = 0, 0, 0, 0

        if epoch in decay_lrs:
            lr = lr * gamma
            adjust_lr(optimizer, lr)
            print('adjusting learning rate to {}'.format(lr))

        for step in range(iter_per_epoch):
            im_data, gt_boxes, im_info = next(train_data_iter)

            if args.use_gpu:
                im_data = im_data.cuda()
                gt_boxes = gt_boxes.cuda()
                im_info = im_info.cuda()

            im_data_variable = Variable(im_data)

            outputs = model(im_data_variable, gt_boxes, im_info)
            rois, _, _, faster_rcnn_cls_loss, faster_rcnn_reg_loss, \
            rpn_cls_loss, rpn_reg_loss, _train_info = outputs

            loss = faster_rcnn_cls_loss.mean() + faster_rcnn_reg_loss.mean() + \
                   rpn_cls_loss.mean() + rpn_reg_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            temp_loss += loss.item()

            if cfg.VERBOSE:
                rpn_tp += _train_info['rpn_tp']
                rpn_tn += _train_info['rpn_tn']
                rpn_fg += _train_info['rpn_num_fg']
                rpn_bg += _train_info['rpn_num_bg']
                faster_rcnn_tp += _train_info['faster_rcnn_tp']
                faster_rcnn_tn += _train_info['faster_rcnn_tn']
                faster_rcnn_fg += _train_info['faster_rcnn_num_fg']
                faster_rcnn_bg += _train_info['faster_rcnn_num_bg']

            if (step + 1) % args.display_interval == 0:
                end_time = time.time()
                temp_loss /= args.display_interval
                rpn_cls_loss_m = rpn_cls_loss.mean().item()
                rpn_reg_loss_m = rpn_reg_loss.mean().item()
                faster_rcnn_cls_loss_m = faster_rcnn_cls_loss.mean().item()
                faster_rcnn_reg_loss_m = faster_rcnn_reg_loss.mean().item()

                print('[epoch %2d][step %4d/%4d] loss: %.4f, time_cost: %.1f' % (epoch, step+1, iter_per_epoch, temp_loss, end_time-start_time))
                print('loss: rpn_cls_loss_m: %.4f, rpn_reg_loss_m: %.4f, faster_rcnn_cls_loss_m: %.4f, faster_rcnn_reg_loss_m: %.4f' %
                      (rpn_cls_loss_m, rpn_reg_loss_m, faster_rcnn_cls_loss_m, faster_rcnn_reg_loss_m))

                if args.use_tfboard:
                    n_iter = epoch * iter_per_epoch + step + 1
                    writer.add_scalar('losses/loss', temp_loss, n_iter)
                    writer.add_scalar('losses/rpn_cls_loss_m', rpn_cls_loss_m, n_iter)
                    writer.add_scalar('losses/rpn_reg_loss_m', rpn_reg_loss_m, n_iter)
                    writer.add_scalar('losses/faster_rcnn_cls_loss_m', faster_rcnn_cls_loss_m, n_iter)
                    writer.add_scalar('losses/faster_rcnn_reg_loss_m', faster_rcnn_reg_loss_m, n_iter)

                    if cfg.VERBOSE:
                        writer.add_scalar('rpn/fg_acc', float(rpn_tp) / rpn_fg, n_iter)
                        writer.add_scalar('rpn/bg_acc', float(rpn_tn) / rpn_bg, n_iter)
                        writer.add_scalar('rcnn/fg_acc', float(faster_rcnn_tp) / faster_rcnn_fg, n_iter)
                        writer.add_scalar('rcnn/bg_acc', float(faster_rcnn_tn) / faster_rcnn_bg, n_iter)

                temp_loss = 0
                rpn_tp, rpn_tn, rpn_fg, rpn_bg = 0, 0, 0, 0
                faster_rcnn_tp, faster_rcnn_tn, faster_rcnn_fg, faster_rcnn_bg = 0, 0, 0, 0
                start_time = time.time()

        if epoch % args.save_interval == 0:
            save_name = os.path.join(args.output_dir, 'faster_rcnn101_epoch_{}.pth'.format(epoch))
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'lr': lr
            }, save_name)


if __name__ == '__main__':
    train()