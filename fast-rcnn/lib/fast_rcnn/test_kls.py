# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms
import cPickle
import heapq
from utils.blob_kls import im_list_to_blob
import os
from skimage.transform import rotate

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    ims = []
    ims.append(im_orig)
    # Create a blob to hold the input images
    blob = im_list_to_blob(ims)

    return blob

def _get_rois_blob(kls):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    kls_blob = np.hstack((np.zeros((kls.shape[0], 1)), kls))
    return kls_blob.astype(np.float32, copy=False)

def _get_blobs(im, kls):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'] = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(kls)
    return blobs

def im_detect(net, im, boxes):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    # if cfg.DEDUP_BOXES > 0:
    #     v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    #     hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
    #     _, index, inv_index = np.unique(hashes, return_index=True,
    #                                     return_inverse=True)
    #     blobs['rois'] = blobs['rois'][index, :]
    #     boxes = boxes[index, :]

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs['rois'].astype(np.float32, copy=False))

    scores = blobs_out['cls_prob']

    # if cfg.DEDUP_BOXES > 0:
    #     # Map scores and predictions back to the original set of boxes
    #     scores = scores[inv_index, :]
    #     pred_boxes = pred_boxes[inv_index, :]

    return scores, boxes

def vis_detections(im, scores, boxes, class_names):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    boxes_num = boxes.shape[0]
    im = im.astype(np.uint8)
    for i in xrange(boxes_num):
        bbox = boxes[i, :4]
        score = scores[i, :]
        label = score.argmax()
        label_score = score[label]
        plt.cla()
        plt.imshow(im, cmap='gray')
        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='g', linewidth=3)
            )
        plt.title('{}  {:.3f}'.format(class_names[label], label_score))
        plt.show()

def test_net(net, klsdb):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(klsdb)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for i in xrange(600, num_images):
        kls_i = klsdb[i]
        imgFile_i = kls_i['image']
        angle = kls_i['angle']
        im = cv2.imread(imgFile_i, 0)
        im = rotate(im, angle, preserve_range=True)
        boxes = kls_i['bbox']
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, boxes)
        _t['im_detect'].toc()
        class_names = ['background', 'arc', 'drapery', 'radial', 'hot-spot']
        vis_detections(im, scores, boxes, class_names)
        pass

