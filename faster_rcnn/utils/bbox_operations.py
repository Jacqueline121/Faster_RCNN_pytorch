import torch


def bbox_overlaps(boxes, query_boxes):

    N = boxes.size(0)
    K = query_boxes.size(0)

    boxes_area = ((boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)).view(N, 1)
    query_boxes_area = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1).view(1, K)

    ix1 = torch.max(boxes[:, 0].view(N, 1), query_boxes[:, 0].view(1, K))
    iy1 = torch.max(boxes[:, 1].view(N, 1), query_boxes[:, 1].view(1, K))
    ix2 = torch.min(boxes[:, 2].view(N, 1), query_boxes[:, 2].view(1, K))
    iy2 = torch.min(boxes[:, 3].view(N, 1), query_boxes[:, 3].view(1, K))

    iw = torch.max(ix2 - ix1 + 1, boxes.new(1).fill_(0))
    ih = torch.max(iy2 - iy1 + 1, boxes.new(1).fill_(0))

    inter_area = iw * ih
    union_area = boxes_area + query_boxes_area - inter_area

    IoU = inter_area / union_area

    return IoU


def bbox_transform(ex_boxes, gt_boxes):
    '''
    :param ex_boxes: expected box (x1, y1, x2, y2)
    :param gt_boxes: ground truth box (x1, y1, x2, y2)
    :return: transforms
    '''

    ex_boxes_xywh = xxyy2xywh(ex_boxes)
    gt_boxes_xywh = xxyy2xywh(gt_boxes)

    t_ctr_x = (gt_boxes_xywh[:, 0] - ex_boxes_xywh[:, 0]) / ex_boxes_xywh[:, 2]
    t_ctr_y = (gt_boxes_xywh[:, 1] - ex_boxes_xywh[:, 1]) / ex_boxes_xywh[:, 3]
    t_w = torch.log(gt_boxes_xywh[:, 2] / ex_boxes_xywh[:, 2])
    t_h = torch.log(gt_boxes_xywh[:, 3] / ex_boxes_xywh[:, 3])

    t_ctr_x = t_ctr_x.view(-1, 1)
    t_ctr_y = t_ctr_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    transform = torch.cat([t_ctr_x, t_ctr_y, t_w, t_h], dim=1)

    return transform


def bbox_transform_inv(bbox, deltas):

    bbox_xywh = xxyy2xywh(bbox)

    pred_ctr_x = deltas[:, 0] * bbox_xywh[:, 2] + bbox_xywh[:, 0]
    pred_ctr_y = deltas[:, 1] * bbox_xywh[:, 3] + bbox_xywh[:, 1]
    pred_w = torch.exp(deltas[:, 2]) * bbox_xywh[:, 2]
    pred_h = torch.exp(deltas[:, 3]) * bbox_xywh[:, 3]

    pred_ctr_x = pred_ctr_x.view(-1, 1)
    pred_ctr_y = pred_ctr_y.view(-1, 1)
    pred_w = pred_w.view(-1, 1)
    pred_h = pred_h.view(-1, 1)

    pred_box = torch.cat([pred_ctr_x, pred_ctr_y, pred_w, pred_h], dim=1)

    return xywh2xxyy(pred_box)


def bbox_transform_inv_cls(boxes, deltas):
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_info):

    boxes[:, 0::4].clamp_(0, im_info[1]-1)
    boxes[:, 1::4].clamp_(0, im_info[0]-1)
    boxes[:, 2::4].clamp_(0, im_info[1]-1)
    boxes[:, 3::4].clamp_(0, im_info[0]-1)

    return boxes


def clip_boxes_cls(boxes, im_shape):

    boxes[:,0::4].clamp_(0, im_shape[1]-1)
    boxes[:,1::4].clamp_(0, im_shape[0]-1)
    boxes[:,2::4].clamp_(0, im_shape[1]-1)
    boxes[:,3::4].clamp_(0, im_shape[0]-1)

    return boxes


def xxyy2xywh(box):
    w = box[:, 2] - box[:, 0] + 1
    h = box[:, 3] - box[:, 1] + 1
    ctr_x = box[:, 0] + w / 2
    ctr_y = box[:, 1] + h / 2

    ctr_x = ctr_x.view(-1, 1)
    ctr_y = ctr_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    xywh_box = torch.cat([ctr_x, ctr_y, w, h], dim=1)
    return xywh_box


def xywh2xxyy(box):
    x1 = box[:, 0] - (box[:, 2] - 1) / 2
    y1 = box[:, 1] - (box[:, 3] - 1) / 2
    x2 = box[:, 0] + (box[:, 2] - 1) / 2
    y2 = box[:, 1] + (box[:, 3] - 1) / 2

    x1 = x1.view(-1, 1)
    y1 = y1.view(-1, 1)
    x2 = x2.view(-1, 1)
    y2 = y2.view(-1, 1)

    xxyy_box = torch.cat([x1, y1, x2, y2], dim=1)
    return xxyy_box







