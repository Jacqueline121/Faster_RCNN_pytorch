"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import cv2
import PIL
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from config.config import cfg
from dataset.factory import get_imdb
import dataset
from dataset.minibatch import get_minibatch


class RoiDataset(Dataset):
    def __init__(self, roidb):
        super(RoiDataset, self).__init__()
        self._roidb = roidb

    def __getitem__(self, i):
        blobs = get_minibatch([self._roidb[i]])
        im_data = blobs['data'][0]
        gt_boxes = blobs['gt_boxes']
        im_info = blobs['im_info'][0]
        return torch.from_numpy(im_data).permute(2, 0, 1), torch.from_numpy(gt_boxes), torch.from_numpy(im_info)

    def __len__(self):
        return len(self._roidb)


def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
             for i in range(imdb.num_images)]
    roidb = imdb.roidb
    for i in range(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def combined_roidb(imdb_names, training=True):
    """
    Combine multiple roidbs
    """

    def get_training_roidb(imdb):
        """Returns a roidb (Region of Interest database) for use in training."""
        if cfg.TRAIN.USE_FLIPPED:
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_images()
            print('done')

        print('Preparing training data...')

        prepare_roidb(imdb)
        # ratio_index = rank_roidb_ratio(imdb)
        print('done')

        return imdb.roidb

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]

    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = get_imdb(imdb_names.split('+')[0])
        # imdb = dataset.imdb.imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)

    if training:
        roidb = filter_roidb(roidb)

    return imdb, roidb


def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
      if len(roidb[i]['boxes']) == 0:
        del roidb[i]
        i -= 1
      i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb



