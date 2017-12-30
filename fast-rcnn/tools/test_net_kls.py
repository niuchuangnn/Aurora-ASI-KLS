#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test_kls import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.get_aurora_klsdb import get_aurora_klsdb
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    prototxt = '/home/amax/NiuChuang/KLSA-auroral-images/fast-rcnn/models/VGG_CNN_M_1024/test_kls.prototxt'
    caffemodel = '/home/amax/NiuChuang/KLSA-auroral-images/Data/region_classification/output/vgg_cnn_m_1024_fast_rcnn_F_iter_50000.caffemodel'
    gpu_id = 1

    bbox_path = '/home/amax/NiuChuang/KLSA-auroral-images/Data/type4_b500_SR_100_440_bbox.hdf5'
    dataFolder = '/home/amax/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/'
    imgType = '.bmp'

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    klsdb = get_aurora_klsdb(bbox_path, dataFolder, imgType)
    test_net(net, klsdb)
