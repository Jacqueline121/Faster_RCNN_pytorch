# Faster_RCNN_pytorch

A Python3.5/Pytroch implementation of Faster RCNN:[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497). And the official implementations are available [here](https://github.com/ShaoqingRen/faster_rcnn). Besides, special thanks for those two repositories：
* [tf-Faster-RCNN](https://github.com/kevinjliang/tf-Faster-RCNN)
* [faster-rcnn.pytorch](https://github.com/tztztztztz/faster-rcnn.pytorch)

### Prerequisites
* python 3.5.x
* pytorch 0.4.1
* tensorboardX
* pillow
* scipy
* numpy
* opencv3
* matplotlib
* easydict

### Results
#### mAP
Train on voc07trainval+voc12trainval
Test on voc07test

| Paper/ResNet101 | This/ResNet101 | This/ResNet50 |
| :-: | :-: | :-: |
| 76.4 | 77.1 | 75.3 |

#### Acc and Loss
The training accuracy of rpn and faster_rcnn:
![Alt text](/result/rpn_acc.png)
![Alt text](/result/faster_rcnn_acc.png)

The training loss curves:
![Alt text](/result/loss.png)

#### Detection Results
![Alt text](/result/result.png)

### Installation

1. Clone this repository (Faster_RCNN_pytorch):
    
        git clone --recursive https://github.com/kevinjliang/tf-Faster-RCNN.git

2. Install dependencies:
    
        cd Faster_RCNN_pytorch
        pip install -r requirements.txt

3. Compile roi_pooling and nms:
    
        cd Faster_RCNN_pytorch/faster_rcnn
        sh make.sh

### Repo Organization
* config: define configuration information of Faster RCNN
* dataset: Scripts for creating, downloading, organizing datasets.
* faster_rcnn: Neural networks and components that form parts of Faster RCNN
* utils: tools package, containing some necessary functions.

### Train

#### Download PASCAL VOC data

1. Download the training, validation, test data:
    
        # download 2007 data
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

        # download 2012 data
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

2. Extract data into one directory named VOCdevkit
    
        # 2007 data
        tar xvf VOCtrainval_06-Nov-2007.tar
        tar xvf VOCtest_06-Nov-2007.tar
        tar xvf VOCdevkit_08-Jun-2007.tar

        # 2012 data
        tar xvf VOCtrainval_11-May-2012.tar

3. It should have this basic structure:
    
        $VOCdevkit/                           # development kit
        $VOCdevkit/VOCcode/                   # VOC utility code
        $VOCdevkit/VOC2007                    # image sets, annotations, etc.
        # ... and several other directories ...

4. Create symlinks for the PASCAL VOC dataset:
    
        cd Faster_RCNN_pytorch/dataset
        mkdir data
        cd data
        # 2007 data
        mkdir VOCdevkit2007
        cd VOCdevkit2007
        ln -s $VOCdevit/VOC2007 VOC2007

        # 2012 data
        mkdir VOCdevkit2012
        cd VOCdevkit2012
        ln -s $VOCdevit/VOC2012 VOC2012

#### Download pretrained ImageNet model
    cd Faster_RCNN_pytorch/faster_rcnn/backbone
    mkdir pretrained
    cd pretrained
    # resnet50-caffe
    wget https://drive.google.com/open?id=0B7fNdx_jAqhtbllXbWxMVEdZclE

    # resnet101-caffe
    wget https://drive.google.com/open?id=0B7fNdx_jAqhtaXZ4aWppWV96czg

#### Train
    python train.py

### Test
    python test.py

If you want to visualize the detection result, you can use:
    
    python test.py --vis



