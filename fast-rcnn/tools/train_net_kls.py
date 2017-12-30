#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.get_aurora_klsdb import get_aurora_klsdb
from fast_rcnn.train_kls import train_net_kls
import caffe
import argparse
import pprint
import numpy as np
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
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
    bbox_path = '/home/amax/NiuChuang/KLSA-auroral-images/Data/type4_b500_SR_100_440_bbox.hdf5'
    data_folder = '/home/amax/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/'
    imgType = '.bmp'
    klsdb = get_aurora_klsdb(bbox_path, data_folder, imgType)
    output_dir = '/home/amax/NiuChuang/KLSA-auroral-images/Data/region_classification/output'

    solver_prototxt = '/home/amax/NiuChuang/KLSA-auroral-images/fast-rcnn/models/VGG_CNN_M_1024/solver_kls.prototxt'
    pretrained_model = '/home/amax/NiuChuang/KLSA-auroral-images/Data/region_classification/output/aurora_iter_10000.caffemodel'
    max_iters = 50000

    train_net_kls(solver_prototxt, klsdb, output_dir,
              pretrained_model=pretrained_model,
              max_iters=max_iters)
