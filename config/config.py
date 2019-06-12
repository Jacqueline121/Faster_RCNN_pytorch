import numpy as np
from easydict import EasyDict

__C = EasyDict()

cfg = __C

__C.DEBUG = False

__C.VERBOSE = True

__C.PRETRAINED_RPN = True

__C.USE_GPU_NMS = True

__C.TRAIN = EasyDict()

__C.TEST = EasyDict()



####################################
# Network
####################################

__C.POOLING_SIZE = 7

####################################
# TRAIN
####################################
__C.TRAIN.BATCH_SIZE = 128

__C.TRAIN.FG_FRACTION = 0.25

__C.TRAIN.FG_THRESH = 0.5
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

__C.TRAIN.TRUNCATED = False
__C.TRAIN.USE_FLIPPED = True
__C.TRAIN.DOUBLE_BIAS = True
__C.TRAIN.SCALES = (600,)
__C.PIXEL_MEANS = np.array([[[103.939, 116.779, 123.68]]])
__C.TRAIN.MAX_SIZE = 1000
__C.TRAIN.USE_ALL_GT = True

__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


####################################
# RPN network
####################################

__C.RPN_ANCHOR_SCALES = np.array([8, 16, 32])
__C.RPN_ANCHOR_RATIOS = [0.5, 1, 2]
__C.FEAT_STRIDE = 16

__C.TRAIN.RPN_CLOBBER_POSITIVE = False

__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

__C.TRAIN.RPN_BATCHSIZE = 256

__C.TRAIN.RPN_FG_FRACTION = 0.5

__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
__C.TRAIN.RPN_NMS_THRESH = 0.7
__C.TRAIN.RPN_MIN_SIZE = 16


__C.TEST.RPN_PRE_NMS_TOP_N = 6000
__C.TEST.RPN_POST_NMS_TOP_N = 300
__C.TEST.RPN_NMS_THRESH = 0.7
__C.TEST.RPN_MIN_SIZE = 16

