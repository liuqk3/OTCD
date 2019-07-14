# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from scipy.misc import imread
from lib.model.utils.config import cfg
from lib.model.utils.blob_single import *
import pdb


def get_minibatch(roidb, num_classes, target_size, im_type=['im', 'mv', 'residual']):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, target_size, im_type=im_type)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)

    blobs['img_id'] = roidb[0]['img_id']

    return blobs


def _get_image_blob(roidb, target_size, im_type=['im', 'residual', 'mv']):
    """Builds an input blob from the images in the roidb at the specified
  scales.
  """
    num_images = len(roidb)

    processed_ims = []
    im_scales = []
    for i in range(num_images):

        import os
        import coviar
        frame_path = roidb[i]['image']  # '/data0/liuqk/MOTChallenge/2DMOT2015/train/ETH-Bahnhof/img1/000229.jpg'

        if frame_path == '/data0/liuqk/MOTChallenge/2DMOT2015/train/KITTI-17/img1/000145.jpg':
            frame_path = '/data0/liuqk/MOTChallenge/2DMOT2015/train/KITTI-17/img1/000144.jpg'
        frame_path_info = frame_path.split('/')

        if 'motchallenge' in roidb[i]['dataset_name']:
            # ['', 'data0', 'liuqk', 'MOTChallenge', '2DMOT2015', 'train', 'ETH-Bahnhof', 'img1', '000229.jpg']

            # get seq_path
            seq_path = '/'
            for j in range(len(frame_path_info) - 2):
                seq_path = os.path.join(seq_path, frame_path_info[j])
            # get the video path
            video_path = os.path.join(seq_path, frame_path_info[-3] + '.mp4')

        elif 'citypersons' in roidb[i]['dataset_name']:
            # ['', 'data0', 'liuqk', 'Cityscapes', 'citysacpesdataset', 'leftImg8bit',
            #  'train', 'tubingen', 'tubingen_000082_000019_leftImg8bit', '000001.png']

            # get seq_path
            seq_path = '/'
            for j in range(len(frame_path_info) - 1):
                seq_path = os.path.join(seq_path, frame_path_info[j])
            # get the video path
            video_path = os.path.join(seq_path, frame_path_info[-2] + '.mp4')

        if not os.path.exists(video_path):
            raise RuntimeError(video_path + ' does not exists')
        frame_id = int(frame_path_info[-1][0:6])

        gop_idx = int((frame_id - 1) / 12)  # GOP starts from 0, while frame_id  here starts from 1.
        in_group_idx = int((frame_id - 1) % 12)  # the index in the group

        if 'im' in im_type:
            im = coviar.load(video_path, gop_idx, in_group_idx, 0, True)
            if len(im.shape) == 2:
                im = im[:, :, np.newaxis]
                im = np.concatenate((im, im, im), axis=2)
            im, im_scale = prep_im_for_blob(im=im,
                                            pixel_normal_scale=cfg.PIXEL_NORMAL_SCALE,
                                            pixel_means=cfg.PIXEL_MEANS,
                                            pixel_stds=cfg.PIXEL_STDS,
                                            target_size=target_size[i],
                                            channel=cfg.PIXEL_CHANNEL)
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]

            im_shape = im.shape
        else:
            im = None

        if 'mv' in im_type:
            mv = coviar.load(video_path, gop_idx, in_group_idx, 1, True)
            mv, im_scale = prep_mv_for_blob(im=mv,
                                            mv_normal_scale=cfg.MV_NORMAL_SCALE,
                                            mv_means=cfg.MV_MEANS,
                                            mv_stds=cfg.MV_STDS,
                                            target_size=target_size[i],
                                            channel=cfg.MV_CHANNEL)
            if roidb[i]['flipped']:
                mv = mv[:, ::-1, :]
                mv[:, :, 0] = - mv[:, :, 0]

            im_shape = mv.shape

        else:
            mv = None

        if 'residual' in im_type:
            residual = coviar.load(video_path, gop_idx, in_group_idx, 2, True)
            # check whether it is a gray image
            if len(residual.shape) == 2:
                residual = residual[:, :, np.newaxis]
                residual = np.concatenate((residual, residual, residual), axis=2)

            residual, im_scale = prep_residual_for_blob(im=residual,
                                                        pixel_normal_scale=cfg.RESIDUAL_NORMAL_SCALE,
                                                        pixel_means=cfg.RESIDUAL_MEANS,
                                                        pixel_stds=cfg.RESIDUAL_STDS,
                                                        target_size=target_size[i],
                                                        channel=cfg.RESIDUAL_CHANNEL)

            if roidb[i]['flipped']:
                residual = residual[:, ::-1, :]

            im_shape = residual.shape
        else:
            residual = None

        im_data = np.zeros((im_shape[0], im_shape[1], 3+2+3))
        if im is not None:
            im_data[:,:,0:3] = im
        if mv is not None:
            im_data[:,:,3:5] = mv
        if residual is not None:
            im_data[:,:,5:8] = residual

        # # ------------ show some results ------------------------
        # from lib.model.utils.misc import show_compressed_frame
        # show_compressed_frame(np.array(im_data[:,:,0:3] + cfg.PIXEL_MEANS, dtype=np.uint8), 0)
        # show_compressed_frame(im_data[:, :, 3:5], 1)
        # show_compressed_frame(np.array(im_data[:, :, 5:8], dtype=np.uint8), 2)

        im_scales.append(im_scale)
        processed_ims.append(im_data)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
