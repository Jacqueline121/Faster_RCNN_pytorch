import torch
import numpy as np
from config.config import cfg
from .generate_anchors import generate_anchors
from ..utils.bbox_operations import bbox_transform_inv, clip_boxes
from ..nms.nms_wrapper import nms


def proposal(rpn_cls_prob, rpn_reg, im_info, train_mode):
    """
    generate proposals

    :param rpn_cls_prob: tensor of shape(B, 2 * A, H, W)
    :param rpn_reg: tensor of shape(B, 4 * A, H, W)
    :param im_info: tensor of shape(B, 3)
    :param train_mode: bool
    :return: rois: tensor of shape(N, 5)
    """

    batch_size, _, height, width = rpn_cls_prob.size()

    assert batch_size == 1, 'only support single batch'

    im_info = im_info[0]

    if train_mode:
        pre_nms_top_n = cfg.TRAIN.RPN_PRE_NMS_TOP_N
        post_nms_top_n = cfg.TRAIN.RPN_POST_NMS_TOP_N
        nms_thresh = cfg.TRAIN.RPN_NMS_THRESH
        min_size = cfg.TRAIN.RPN_MIN_SIZE
    else:
        pre_nms_top_n = cfg.TEST.RPN_PRE_NMS_TOP_N
        post_nms_top_n = cfg.TEST.RPN_POST_NMS_TOP_N
        nms_thresh = cfg.TEST.RPN_NMS_THRESH
        min_size = cfg.TEST.RPN_MIN_SIZE

    anchor_scales = cfg.RPN_ANCHOR_SCALES
    anchor_ratios = cfg.RPN_ANCHOR_RATIOS
    feat_stride = cfg.FEAT_STRIDE

    # generate anchors
    _anchors = generate_anchors(base_size=feat_stride, scales=anchor_scales, ratios=anchor_ratios)
    num_anchors = _anchors.shape[0]

    A = num_anchors
    K = int(height * width)
    shift_x = np.arange(0, width) * cfg.FEAT_STRIDE
    shift_y = np.arange(0, height) * cfg.FEAT_STRIDE
    shifts_x, shifts_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shifts_x.ravel(), shifts_y.ravel(), shifts_x.ravel(), shifts_y.ravel())).transpose()

    all_anchors = _anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
    all_anchors = all_anchors.reshape(-1, 4)
    num_all_anchors = all_anchors.shape[0]

    assert num_all_anchors == K * A

    all_anchors = torch.from_numpy(all_anchors).type_as(rpn_cls_prob)

    rpn_reg = rpn_reg.permute(0, 2, 3, 1).contiguous().view(-1, 4)

    # generate all rois
    proposals = bbox_transform_inv(all_anchors, rpn_reg)
    proposals = clip_boxes(proposals, im_info)

    # filter proposals
    keep_inds = _filter_proposal(proposals, min_size)

    proposals_keep = proposals[keep_inds, :]

    # proposal prob
    proposals_prob = rpn_cls_prob[:, num_anchors:, :, :]
    proposals_prob = proposals_prob.permute(0, 2, 3, 1).contiguous().view(-1)

    proposals_prob_keep = proposals_prob[keep_inds]

    # sort prob
    order = torch.sort(proposals_prob_keep, descending=True)[1]

    top_keep = order[:pre_nms_top_n]

    proposals_keep = proposals_keep[top_keep, :]
    proposals_prob_keep = proposals_prob_keep[top_keep]

    # nms
    keep = nms(torch.cat([proposals_keep, proposals_prob_keep.view(-1, 1)], dim=1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
    keep = keep.long().view(-1)
    top_keep = keep[:post_nms_top_n]

    proposals_keep = proposals_keep[top_keep, :]
    proposals_prob_keep = proposals_prob_keep[top_keep]

    rois = proposals_keep.new_zeros((proposals_keep.size(0), 5))

    rois[:, 1:] = proposals_keep

    return rois


def _filter_proposal(proposals, min_size):

    width = proposals[:, 2] - proposals[:, 0] + 1
    height = proposals[:, 3] - proposals[:, 1] + 1

    keep_inds = ((width >= min_size) & (height >= min_size))

    return keep_inds











