import torch
import numpy as np
from config.config import cfg
from ..utils.bbox_operations import bbox_overlaps, bbox_transform


def proposal_target(rois, gt_boxes, num_classes):
    """
    Assign ground-truth targets for rois, including classification label targets and
    bounding-box regression targets.

    :param rois: tensor of shape (N, 5), [0, x1, y1, x2, y2]
    :param gt_boxes: tensor of shape (B, G, 5), [x1, y1, x2, y2, class_label]
    :param num_classes: 21
    :return:
    rois: tensor of shape (S, 5)
    label_target: tensor of shape (S)
    boxes_target: tensor of shape (S, 4 * n_classes)
    boxes_inside_weights: tensor of shape (S, 4 * n_classes)
    boxes_outside_weights: tensor of shape (S, 4 * n_classes)

    """

    batch_size, num_gt_boxes, _ = gt_boxes.size()

    assert batch_size == 1, 'only support single batch'

    gt_boxes = gt_boxes[0]
    gt_boxes_append = rois.new_zeros((num_gt_boxes, 5))
    gt_boxes_append[:, 1:] = gt_boxes[:, :4]

    merged_rois = torch.cat([rois, gt_boxes_append], dim=0)

    # calculate IoU
    rois_gt_overlap = bbox_overlaps(merged_rois[:, 1:], gt_boxes[:, :4])
    rois_gt_max_overlap, rois_gt_argmax_overlap = torch.max(rois_gt_overlap, dim=1)

    # label target
    label_targets = gt_boxes[rois_gt_argmax_overlap, 4].long()

    fg_inds = torch.nonzero(rois_gt_max_overlap >= cfg.TRAIN.FG_THRESH).view(-1)

    bg_inds = torch.nonzero((rois_gt_max_overlap >= cfg.TRAIN.BG_THRESH_LO) &
                            (rois_gt_max_overlap < cfg.TRAIN.BG_THRESH_HI)).view(-1)

    fg_num = fg_inds.numel()
    bg_num = bg_inds.numel()

    rois_per_image = cfg.TRAIN.BATCH_SIZE
    fg_rois_per_image = int(rois_per_image * cfg.TRAIN.FG_FRACTION)

    if bg_num > 0:
        fg_rois_this_image = min(fg_num, fg_rois_per_image)
        rand_num = torch.from_numpy(np.random.permutation(fg_rois_this_image)).type_as(rois).long()
        fg_inds = fg_inds[rand_num[:fg_rois_this_image]]

        bg_rois_this_image = rois_per_image - fg_rois_this_image
        rand_num = torch.from_numpy(np.floor(np.random.rand(bg_rois_this_image) * bg_num)).type_as(gt_boxes).long()
        bg_inds = bg_inds[rand_num]
    else:
        rand_num = torch.from_numpy(np.floor(np.random.rand(rois_per_image) * fg_num)).type_as(gt_boxes).long()
        fg_inds = fg_inds[rand_num]
        fg_rois_this_image = rois_per_image
        bg_rois_this_image = 0

    keep_inds = torch.cat([fg_inds, bg_inds], dim=0)
    label_targets = label_targets[keep_inds]

    if bg_rois_this_image > 0:
        label_targets[fg_rois_this_image:] = 0

    rois = merged_rois[keep_inds, :]

    # bbox target
    bbox_deltas = bbox_transform(rois[:, 1:], gt_boxes[rois_gt_argmax_overlap[keep_inds], :4])

    bbox_normalize_means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).type_as(rois).view(1, 4)
    bbox_normalize_stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).type_as(rois).view(1, 4)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
        bbox_deltas = (bbox_deltas - bbox_normalize_means) / bbox_normalize_stds

    num_bbox = rois.size(0)
    bbox_target = rois.new_zeros(num_bbox, 4 * num_classes)
    bbox_inside_weights = rois.new_zeros(num_bbox, 4 * num_classes)
    inside_weights = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS).type_as(rois)

    for ind in range(num_bbox):
        label = int(label_targets[ind].item())
        if label > 0:
            start = label * 4
            end = start + 4
            bbox_target[ind, start:end] = bbox_deltas[ind, :]
            bbox_inside_weights[ind, start:end] = inside_weights

    bbox_outside_weights = (bbox_inside_weights > 0).float()

    return rois, label_targets, bbox_target, bbox_inside_weights, bbox_outside_weights






