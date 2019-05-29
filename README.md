# Faster_RCNN_pytorch

I re-implemented Faster R-CNN

### Prerequisites
* python 3.5.x
* pytorch 0.4.1
* tensorboardX
* pillow
* numpy
* opencv3
* matplotlib
* easydict


### Installation

1. Clone this repository (Faster_RCNN_pytorch):
$ git clone --recursive https://github.com/kevinjliang/tf-Faster-RCNN.git

2. Install dependencies:
$ cd $Faster_RCNN_pytorch
$ pip install -r requirements.txt

3. Compile roi_pooling and nms:
$ cd $Faster_RCNN_pytorch/faster_rcnn
$ sh make.sh

### Training on PASCAL VOC

#### Train
    $ python train.py

#### Test
    $ python test.py



