# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
from lib.utils.misc import resize_im
import cv2

# from scipy.misc import imread, imresize

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0) # [max_h, max_w]
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], max_shape[2]),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def group_ims_list_to_blob(ims):
    """
     We just add a new batch dim to the image
    data.
    :param ims: a list , contains the image of one group. Each frame of one group is organized as
                a list (sub-list) with the length of 3, and the 1-st 2-nd 3-th of this sub-sub-sub list are
                I frame, motion vector, residual.
    :return:
    """
    for j in range(len(ims)):

        if ims[j][0] is not None:
            ims[j][0] = np.expand_dims(ims[j][0], axis=0)  # I frame

        if ims[j][1] is not None:
            ims[j][1] = np.expand_dims(ims[j][1], axis=0)  # motion vector

        if ims[j][1] is not None:
            ims[j][2] = np.expand_dims(ims[j][2], axis=0)  # residual

    return ims


def prep_im_for_blob(im, pixel_normal_scale, pixel_means, pixel_stds, target_size, channel='BGR'):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)  # the input image default is BRG

    im = im / pixel_normal_scale # resize to [0, 1]
    im -= pixel_means
    im = im / pixel_stds
    # im = im[:, :, ::-1]
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    # im = imresize(im, im_scale)

    # im = resize_im(im, im_scale)

    #im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    # BGR to RGB
    if channel == 'RGB':
        im = im[:, :, ::-1]
    im = resize_im(im, im_scale)

    return im, im_scale

def prep_residual_for_blob(im, pixel_normal_scale, pixel_means, pixel_stds, target_size, channel='BGR'):
    """scale a residual image for use in a blob.
        we do not do mean subtraction for residual.
    """

    im = im.astype(np.float32, copy=False) # the input image default is BRG
    im = im / pixel_normal_scale # resize to [0, 1]
    im = im - pixel_means
    im = im / pixel_stds

    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    #im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im = resize_im(im_data=im, im_scale=im_scale)

    # BGR to RGB
    if channel == 'RGB':
        im = im[:, :, ::-1]
    return im, im_scale


def prep_mv_for_blob(im, mv_normal_scale, mv_means, mv_stds, target_size, channel='XY'):
    """scale a motion vector field for use in a blob.
    """
    im = im.astype(np.float32, copy=False) # the default mv is 'xy'

    im = im / mv_normal_scale
    im = im - mv_means
    im = im / mv_stds

    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    # im = imresize(im, im_scale)
    im = resize_im(im_data=im, im_scale=im_scale)

    # the motion vector
    im = im * im_scale

    if channel == 'YX':
        im = im[:, :, ::-1]

    return im, im_scale