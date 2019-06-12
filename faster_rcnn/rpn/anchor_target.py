import numpy as np
import torch
from config.config import cfg
from .generate_anchors import generate_anchors
from ..utils.bbox_operations import bbox_overlaps, bbox_transform


def anchor_target(rpn_cls_prob, gt_boxes, im_info):
    """
    Assign ground-truth targets for anchors, including classification label targets and
    bounding-box regression targets.

    :param rpn_cls_prob: tensor of shape(B, A*2, H, W)
    :param gt_boxes: tensor of shape(B, N, 5)
    :param im_info: tensor of shape(B, 3)
    :return:
    rpn_label_target: tensor of shape(B, A*2, H, W)
    rpn_bbox_target: tensor of shape(B, A*4, H, W)
    rpn_bbox_inside_weights: tensor of shape(B, A*4, H, W)
    rpn_bbox_outside_weights: tensor of shape(B, A*4, H, W)
    """

    batch_size, _, height, width = rpn_cls_prob.size()
    gt_boxes = gt_boxes[0]
    num_gt_boxes = gt_boxes.size(0)
    im_info = im_info[0]
    im_height, im_width = im_info[0], im_info[1]
    allowed_border = 0

    anchor_scales = cfg.RPN_ANCHOR_SCALES
    anchor_ratios = cfg.RPN_ANCHOR_RATIOS
    feat_stride = cfg.FEAT_STRIDE

    assert batch_size == 1, 'only support single batch'

    # generate anchors
    _anchors = generate_anchors(base_size=feat_stride, ratios=anchor_ratios, scales=anchor_scales)
    num_anchors = _anchors.shape[0]

    A = num_anchors
    K = height * width
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shifts_x, shifts_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shifts_x.ravel(), shifts_y.ravel(), shifts_x.ravel(), shifts_y.ravel())).transpose()

    all_anchors = _anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
    all_anchors = all_anchors.reshape(-1, 4)
    num_all_anchors = all_anchors.shape[0]

    assert num_all_anchors == A * K

    all_anchors = torch.from_numpy(all_anchors).type_as(rpn_cls_prob)

    # filter outside anchors
    inside_inds = (
        (all_anchors[:, 0] >= -allowed_border) &
        (all_anchors[:, 1] >= -allowed_border) &
        (all_anchors[:, 2] <= im_width + allowed_border - 1) &
        (all_anchors[:, 3] <= im_height + allowed_border - 1)
    )

    inside_inds = torch.nonzero(inside_inds).view(-1)
    inside_anchors = all_anchors[inside_inds, :]
    num_inside_anchors = inside_anchors.size(0)

    overlaps = bbox_overlaps(inside_anchors, gt_boxes[:, :4])
    anchor_gt_max_overlap, anchor_gt_argmax_overlap = torch.max(overlaps, dim=1)
    gt_anchor_max_overlap, gt_anchor_argmax_overlap = torch.max(overlaps, dim=0)

    # label target
    label_target = rpn_cls_prob.new(num_inside_anchors).fill_(-1)

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVE:
        label_target[anchor_gt_max_overlap < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    keep = torch.sum(gt_anchor_max_overlap.view(1, -1).expand(num_inside_anchors, num_gt_boxes) == overlaps, dim=1)
    if torch.sum(keep) > 0:
        label_target[keep > 0] = 1

    label_target[anchor_gt_max_overlap > cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVE:
        label_target[anchor_gt_max_overlap < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    max_fg_num = int(cfg.TRAIN.RPN_BATCHSIZE * cfg.TRAIN.RPN_FG_FRACTION)
    fg_inds = torch.nonzero(label_target == 1).view(-1)
    fg_num = fg_inds.size(0)

    if fg_num > max_fg_num:
        rand_num = torch.from_numpy(np.random.permutation(fg_num)).type_as(fg_inds)
        discard_inds = fg_inds[rand_num[: (fg_num - max_fg_num)]]
        label_target[discard_inds] = -1

    max_bg_num = int(cfg.TRAIN.RPN_BATCHSIZE - torch.sum(label_target == 1))
    bg_inds = torch.nonzero(label_target == 0).view(-1)
    bg_num = bg_inds.size(0)

    if bg_num > max_bg_num:
        rand_num = torch.from_numpy(np.random.permutation(bg_num)).type_as(bg_inds)
        discard_inds = bg_inds[rand_num[: (bg_num - max_bg_num)]]
        label_target[discard_inds] = -1

    # bbox target
    bbox_target = bbox_transform(inside_anchors, gt_boxes[anchor_gt_argmax_overlap, :4])
    bbox_inside_weights = rpn_cls_prob.new_zeros(num_inside_anchors, 4)
    bbox_inside_weights[label_target==1, :] = torch.from_numpy(np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)).type_as(rpn_cls_prob)

    bbox_outside_weights = rpn_cls_prob.new_zeros(num_inside_anchors, 1)
    num_examples = torch.sum(label_target >= 0).float()
    bbox_outside_weights[label_target >= 0, :] = 1.0 / num_examples
    bbox_outside_weights = bbox_outside_weights.expand(num_inside_anchors, 4)

    rpn_label_target = _unmap(label_target, num_all_anchors, inside_inds, -1)
    bbox_target = _unmap(bbox_target, num_all_anchors, inside_inds, 0)
    bbox_inside_weights = _unmap(bbox_inside_weights, num_all_anchors, inside_inds, 0)
    bbox_outside_weights = _unmap(bbox_outside_weights, num_all_anchors, inside_inds, 0)

    rpn_bbox_target = bbox_target.view(batch_size, height, width, 4 * A).permute(0, 3, 1, 2)
    rpn_bbox_inside_weights = bbox_inside_weights.view(batch_size, height, width, 4 * A).permute(0, 3, 1, 2)
    rpn_bbox_outside_weights = bbox_outside_weights.view(batch_size, height, width, 4 * A).permute(0, 3, 1, 2)

    return rpn_label_target, rpn_bbox_target, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):

    if data.dim() == 1:
        ret = data.new(count).fill_(fill)
        ret[inds] = data
    elif data.dim() == 2:
        ret = data.new(count, data.size(1)).fill_(fill)
        ret[inds, :] = data
    else:
        raise NotImplementedError

    return ret












