import os
import argparse
import torch
import numpy as np
from config.config import cfg
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.nms.nms_wrapper import nms
from dataset.roidb import RoiDataset, combined_roidb
from torch.utils.data import DataLoader
from torch.autograd import Variable
from faster_rcnn.utils.bbox_operations import bbox_transform_inv_cls, clip_boxes_cls
from utils.visualize import draw_detection_boxes
import time
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Faster RCNN')
    parser.add_argument('--dataset', dest='dataset', default='voc07test', type=str)
    parser.add_argument('--backbone', dest='backbone', default='resnet101', type=str)
    parser.add_argument('--use_gpu', dest='use_gpu', default=True, type=bool)
    parser.add_argument('--batch_size', dest='batch_size', default=1, type=int)
    parser.add_argument('--thresh', dest='thresh', default=0.0, type=float)
    parser.add_argument('--max_per_image', dest='max_per_image', default=10, type=int)
    parser.add_argument('--check_epoch', dest='check_epoch', default=9, type=int)
    parser.add_argument('--output_dir', dest='output_dir', default='output', type=str)
    parser.add_argument('--vis', dest='vis', default=False, type=bool)

    args = parser.parse_args()
    return args


def test():
    args = parse_args()

    # perpare data
    print('load data')
    if args.dataset == 'voc07test':
        dataset_name = 'voc_2007_test'
    elif args.dataset == 'voc12test':
        dataset_name = 'voc_2012_test'
    else:
        raise NotImplementedError

    imdb, roidb = combined_roidb(dataset_name)
    test_dataset = RoiDataset(roidb)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    test_data_iter = iter(test_dataloader)

    # load model
    model = FasterRCNN(backbone=args.backbone)
    model_name = '0712_faster_rcnn101_epoch_{}.pth'.format(args.check_epoch)
    model_path = os.path.join(args.output_dir, model_name)
    model.load_state_dict(torch.load(model_path)['model'])

    if args.use_gpu:
        model = model.cuda()

    model.eval()

    num_images = len(imdb.image_index)
    det_file_path = os.path.join(args.output_dir, 'detections.pkl')

    all_boxes = [[[] for _ in range(num_images)] for _ in range(imdb.num_classes)]

    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    torch.set_grad_enabled(False)

    for i in range(num_images):
        start_time = time.time()
        im_data, gt_boxes, im_info = next(test_data_iter)
        if args.use_gpu:
            im_data = im_data.cuda()
            gt_boxes = gt_boxes.cuda()
            im_info = im_info.cuda()

        im_data_variable = Variable(im_data)

        det_tic = time.time()
        rois, faster_rcnn_cls_prob, faster_rcnn_reg, _, _, _, _, _train_info = model(im_data_variable, gt_boxes,
                                                                                     im_info)

        scores = faster_rcnn_cls_prob.data
        boxes = rois.data[:, 1:]
        boxes_deltas = faster_rcnn_reg.data

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            boxes_deltas = boxes_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            boxes_deltas = boxes_deltas.view(-1, 4 * imdb.num_classes)

        pred_boxes = bbox_transform_inv_cls(boxes, boxes_deltas)

        pred_boxes = clip_boxes_cls(pred_boxes, im_info[0])

        pred_boxes /= im_info[0][2].item()

        det_toc = time.time()
        detect_time = det_tic - det_toc
        nms_tic = time.time()

        if args.vis:
            im_show = Image.open(imdb.image_path_at(i))

        for j in range(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > args.thresh).view(-1)

            if inds.numel() > 0:
                cls_score = scores[:, j][inds]
                _, order = torch.sort(cls_score, 0, True)
                cls_boxes = pred_boxes[inds][:, j * 4: (j + 1) * 4]
                cls_dets = torch.cat((cls_boxes, cls_score.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, 0.3)
                cls_dets = cls_dets[keep.view(-1).long()]
                if args.vis:
                    cls_name_dets = np.repeat(j, cls_dets.size(0))
                    im_show = draw_detection_boxes(im_show, cls_dets.cpu().numpy(), cls_name_dets, imdb.classes, 0.5)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        if args.max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > args.max_per_image:
                image_thresh = np.sort(image_scores)[-args.max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        if args.vis:
            plt.imshow(im_show)
            plt.show()
        nms_toc = time.time()
        nms_time = nms_tic - nms_toc
        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

    with open(det_file_path, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, args.output_dir)

    end_time = time.time()
    print("test time: %0.4fs" % (end_time - start_time))


if __name__ == '__main__':
    test()