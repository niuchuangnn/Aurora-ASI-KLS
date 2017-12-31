# Weakly Supervised Semantic Semantic Segmentation for Join Key Local Structure Localization and Classification of Aurora Image
By Chuang Niu, Jun Zhang, Qian Wang, and Jimin Liang

## Introduction
TO DO

## Installation

### Requirements (Tested on x64 Unbuntu 14.04 environment)
This project is based on:
* [selective search of python version](https://github.com/BradNeuberg/selective_search_py)
* [fast-rcnn](https://github.com/rbgirshick/fast-rcnn)

It is noted that the original codes of [selective search](https://github.com/BradNeuberg/selective_search_py) and [fast-rcnn](https://github.com/rbgirshick/fast-rcnn)
are not uesd by this project, but you must make sure that they can run normally before implementation of this project.

### Get started
1. Get the code. We will call the diretory that you cloned Aurora-ASI-KLS into `$KLS_ROOT`

```
git clone https://github.com/niuchuangnn/Aurora-ASI-KLS
cd $KLS_ROOT/selective_search_py
wget http://cs.brown.edu/~pff/segment/segment.zip; unzip segment.zip; rm segment.zip
cmake .
make

cd &KLS_ROOT/fast-rcnn/caffe-fast-rcnn
make all
make pycaffe
cd &KLS_ROOT/fast-rcnn/lib
make

```
2. Download the region detection model: [vcc_cnn_m_fast_rcnn_b500_iter_10000.caffemodel](https://1drv.ms/u/s!ArnlNXPnKNAKjQWsM4hsLuvu8cNW)

```
cd $KLS_ROOT/Data
mkdir -p Data/region_classification
```
Put the downloaded model into this folder.

3. Run demo.

```
cd KLS_ROOT/src/demo
python demo.py
```
You will see:
```
classification time: 1.2338631897
segmentation time: 1.77530801296
```
![arc](https://github.com/niuchuangnn/Aurora-ASI-KLS/blob/master/Data/demo_examples/a_r.png)
![drapery](https://github.com/niuchuangnn/Aurora-ASI-KLS/blob/master/Data/demo_examples/D_r.png)
![radial](https://github.com/niuchuangnn/Aurora-ASI-KLS/blob/master/Data/demo_examples/R_r.png)
![hot-spot](https://github.com/niuchuangnn/Aurora-ASI-KLS/blob/master/Data/demo_examples/HS_r.png)
