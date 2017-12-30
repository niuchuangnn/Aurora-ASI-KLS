# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob_kls import prep_im_for_blob, im_list_to_blob
from skimage.transform import rotate

def get_minibatch(klsdb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(klsdb)
    batch_bg_num = cfg.TRAIN.BG_NUM
    batch_fg_num = cfg.TRAIN.FG_NUM


    # Now, build the region of interest and label blobs
    kls_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0), dtype=np.float32)
    # all_overlaps = []
    for im_i in xrange(num_images):
        im = klsdb[im_i]
        kls = im['bbox']
        labels = im['gt_classes']
        batch_ind = im_i * np.ones((kls.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, kls))
        kls_blob = np.vstack((kls_blob, rois_blob_this_image))

        # Add to labels, bbox targets, and bbox loss blobs
        labels_blob = np.hstack((labels_blob, labels))

        # all_overlaps = np.hstack((all_overlaps, overlaps))

    fg_inds = list(np.where(labels_blob > 0)[0])
    bg_inds = list(np.where(labels_blob == 0)[0])

    fg_inds = fg_inds[0:batch_fg_num]
    bg_inds = bg_inds[0:batch_bg_num]
    fg_keep = list(np.array(kls_blob[fg_inds, 0], dtype='i'))
    bg_keep = list(np.array(kls_blob[bg_inds, 0], dtype='i'))
    keep = list(set(fg_keep+bg_keep))
    klsdb_keep = [klsdb[x] for x in range(num_images) if x in keep]
    # Get the input image blob, formatted for caffe
    im_blob = _get_image_blob(klsdb_keep)

    kls_blob = np.vstack((kls_blob[fg_inds, :], kls_blob[bg_inds, :]))
    labels_blob = np.hstack((labels_blob[fg_inds], labels_blob[bg_inds]))

    # update batch id
    for ni in range(len(keep)):
        kls_blob[:, 0][np.where(kls_blob[:, 0]==keep[ni])] = ni

    # For debug visualizations
    # _vis_minibatch(im_blob, kls_blob, labels_blob)

    blobs = {'data': im_blob,
             'rois': kls_blob,
             'labels': labels_blob}

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image,
                             replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image,
                             replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_loss_weights = \
            _get_bbox_regression_labels(roidb['bbox_targets'][keep_inds, :],
                                        num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_loss_weights

def _get_image_blob(klsdb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(klsdb)
    processed_ims = []
    for i in xrange(num_images):
        im = cv2.imread(klsdb[i]['image'], 0)
        angle = klsdb[i]['angle']
        # if roidb[i]['flipped']:
        #     im = im[:, ::-1, :]
        im = rotate(im, angle, preserve_range=True)
        im = prep_im_for_blob(im, cfg.PIXEL_MEANS)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_loss_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]
    return bbox_targets, bbox_loss_weights

def _vis_minibatch(im_blob, kls_blob, labels_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(kls_blob.shape[0]):
        rois = kls_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, 0, :, :].copy()
        im += cfg.PIXEL_MEANS

        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im, cmap='gray')
        print 'class: ', cls
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
