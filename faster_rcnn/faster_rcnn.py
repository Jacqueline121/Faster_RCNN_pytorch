import os
import torch
from config.config import cfg
import torch.nn as nn
import torch.nn.functional as F
from .backbone.resnet import resnet50, resnet101
from .rpn.rpn import RPN
from .rpn.proposal_target import proposal_target
from .roi_pooling.modules.roi_pool import _RoIPooling
from .utils.smooth_L1 import smooth_L1
from .utils.normal_init import normal_init


class FasterRCNN(nn.Module):
    def __init__(self, backbone):
        super(FasterRCNN, self).__init__()

        if backbone == 'resnet101':
            resnet = resnet101()
        elif backbone == 'resnet50':
            resnet = resnet50()
        else:
            raise NotImplementedError

        pretrained = os.path.join('faster_rcnn/backbone', 'pretrained', '{}-caffe.pth'.format(backbone))
        state_dict = torch.load(pretrained)
        resnet.load_state_dict({k: v for k, v in state_dict.items() if k in resnet.state_dict()})

        self.backbone = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                      resnet.layer1, resnet.layer2, resnet.layer3)
        self.rpn = RPN()

        self.pooled_height = cfg.POOLING_SIZE
        self.pooled_width = cfg.POOLING_SIZE
        self.spatial_scale = 1.0 / cfg.FEAT_STRIDE
        self.roi_pooling = _RoIPooling(self.pooled_height, self.pooled_width, self.spatial_scale)

        self.classifier = nn.Sequential(resnet.layer4)
        self.num_classes = 21
        self.faster_rcnn_cls = nn.Linear(2048, self.num_classes)
        self.faster_rcnn_reg = nn.Linear(2048, self.num_classes * 4)

        for p in self.backbone[0].parameters(): p.requires_grad = False
        for p in self.backbone[1].parameters(): p.requires_grad = False
        for p in self.backbone[4].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            class_name = m.__class__.__name__
            if class_name.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.backbone.apply(set_bn_fix)
        self.classifier.apply(set_bn_fix)

        normal_init(self.rpn.rpn_conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn.rpn_cls, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn.rpn_reg, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.faster_rcnn_cls, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.faster_rcnn_reg, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def forward(self, im_data, gt_boxes, im_info):
        # backbone
        features = self.backbone(im_data)

        # rpn
        rois, rpn_cls_loss, rpn_reg_loss, _rpn_train_info = self.rpn(features, gt_boxes, im_info)

        if self.training:
            # calculate proposal target
            rois, label_targets, bbox_targets, bbox_inside_weights, bbox_outside_weights = proposal_target(rois, gt_boxes, self.num_classes)
        else:
            label_targets = None
            bbox_targets = None
            bbox_inside_weights = None
            bbox_outside_weights = None
        
        # roi pooling layer
        rois_pool_feature = self.roi_pooling(features, rois)

        out = self.classifier(rois_pool_feature)
        out = out.mean(3).mean(2)  

        # classification and regression
        faster_rcnn_cls_score = self.faster_rcnn_cls(out)
        faster_rcnn_reg = self.faster_rcnn_reg(out)

        faster_rcnn_cls_prob = F.softmax(faster_rcnn_cls_score, dim=1)

        faster_rcnn_cls_loss = 0
        faster_rcnn_reg_loss = 0
        _train_info = {}

        if self.training:
            # loss
            faster_rcnn_cls_loss = F.cross_entropy(faster_rcnn_cls_score, label_targets)
            faster_rcnn_reg_loss = smooth_L1(faster_rcnn_reg, bbox_targets, bbox_inside_weights, bbox_outside_weights)
            if cfg.VERBOSE:
                _faster_rcnn_fg_inds = torch.nonzero(label_targets != 0).view(-1)
                _faster_rcnn_bg_inds = torch.nonzero(label_targets == 0).view(-1)
                _faster_rcnn_pred_info = torch.argmax(faster_rcnn_cls_score, dim=1)
                _faster_rcnn_tp = torch.sum(_faster_rcnn_pred_info[_faster_rcnn_fg_inds] == label_targets[_faster_rcnn_fg_inds])
                _faster_rcnn_tn = torch.sum(_faster_rcnn_pred_info[_faster_rcnn_bg_inds] == label_targets[_faster_rcnn_bg_inds])
                _train_info['faster_rcnn_num_fg'] = _faster_rcnn_fg_inds.size(0)
                _train_info['faster_rcnn_num_bg'] = _faster_rcnn_bg_inds.size(0)
                _train_info['faster_rcnn_tp'] = _faster_rcnn_tp.item()
                _train_info['faster_rcnn_tn'] = _faster_rcnn_tn.item()
                _train_info['rpn_num_fg'] = _rpn_train_info['rpn_num_fg']
                _train_info['rpn_num_bg'] = _rpn_train_info['rpn_num_bg']
                _train_info['rpn_tp'] = _rpn_train_info['rpn_tp']
                _train_info['rpn_tn'] = _rpn_train_info['rpn_tn']

        return rois, faster_rcnn_cls_prob, faster_rcnn_reg, faster_rcnn_cls_loss, faster_rcnn_reg_loss, rpn_cls_loss, rpn_reg_loss, _train_info

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            self.backbone.eval()
            self.backbone[5].train()
            self.backbone[6].train()

            def set_bn_eval(m):
                class_name = m.__class__.__name__
                if class_name.find('BatchNorm') != -1:
                    m.eval()
            self.backbone.apply(set_bn_eval)
            self.classifier.apply(set_bn_eval)









