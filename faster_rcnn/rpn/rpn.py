import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .anchor_target import anchor_target
from .proposal import proposal
from config.config import cfg
from ..utils.smooth_L1 import smooth_L1


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

        self.anchor_scales = cfg.RPN_ANCHOR_SCALES
        self.anchor_ratios = cfg.RPN_ANCHOR_RATIOS
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

        self.rpn_conv = nn.Conv2d(1024, 512, 3, 1, 1)
        self.rpn_cls = nn.Conv2d(512, self.num_anchors * 2, 1, 1, 0)
        self.rpn_reg = nn.Conv2d(512, self.num_anchors * 4, 1, 1, 0)

    def forward(self, feature, gt_boxes, im_info):
        batch_size, _, height, width = feature.size()

        rpn_features = F.relu(self.rpn_conv(feature), inplace=True)

        # classification and regression
        rpn_cls_score = self.rpn_cls(rpn_features)  
        rpn_reg = self.rpn_reg(rpn_features)

        rpn_cls_score_reshape = rpn_cls_score.view(batch_size, 2, self.num_anchors, height, width)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob = rpn_cls_prob_reshape.view(batch_size, 2 * self.num_anchors, height, width)

        # calculate proposals
        rois = proposal(rpn_cls_prob.data, rpn_reg.data, im_info, self.training)

        rpn_cls_loss = 0
        rpn_reg_loss = 0
        _rpn_train_info = {}
        if self.training:
            # calculate anchor target
            rpn_label_targets, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target(rpn_cls_prob.data, gt_boxes, im_info)

            keep_inds = torch.nonzero(rpn_label_targets != -1).view(-1)
            rpn_label_targets_keep = Variable(rpn_label_targets[keep_inds]).long()

            keep_inds = Variable(keep_inds)

            rpn_cls_score = rpn_cls_score_reshape.permute(0, 3, 4, 2, 1).contiguous().view(-1, 2)

            # cross entropy loss
            rpn_cls_loss = F.cross_entropy(rpn_cls_score[keep_inds, :], rpn_label_targets_keep)

            # smooth L1 loss
            rpn_bbox_targets = Variable(rpn_bbox_targets)
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_reg_loss = smooth_L1(rpn_reg, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, dim=[1, 2, 3])

            if cfg.VERBOSE:
                _rpn_fg_inds = torch.nonzero(rpn_label_targets == 1).view(-1)
                _rpn_bg_inds = torch.nonzero(rpn_label_targets == 0).view(-1)
                _rpn_num_fg = _rpn_fg_inds.size(0)
                _rpn_num_bg = _rpn_bg_inds.size(0)
                _rpn_pred_data = rpn_cls_prob_reshape.permute(0, 3, 4, 2, 1).contiguous().view(-1, 2)[:, 1]
                _rpn_pred_info = (_rpn_pred_data >= 0.4).long()
                _rpn_tp = torch.sum(rpn_label_targets[_rpn_fg_inds].long() == _rpn_pred_info[_rpn_fg_inds])
                _rpn_tn = torch.sum(rpn_label_targets[_rpn_bg_inds].long() == _rpn_pred_info[_rpn_bg_inds])
                _rpn_train_info['rpn_num_fg'] = _rpn_num_fg
                _rpn_train_info['rpn_num_bg'] = _rpn_num_bg
                _rpn_train_info['rpn_tp'] = _rpn_tp.item()
                _rpn_train_info['rpn_tn'] = _rpn_tn.item()

        return rois, rpn_cls_loss, rpn_reg_loss, _rpn_train_info



