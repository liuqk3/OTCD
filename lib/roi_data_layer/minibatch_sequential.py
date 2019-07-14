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
import os
import coviar

import pdb


def get_minibatch(roidb, num_classes, target_size, group_size, im_type):
    """Given a roidb, construct a minibatch sampled from it."""

    # Get the input image blob, formatted for caffe
    im_blob, im_scale = _get_image_blob(roidb, target_size, group_size, im_type)

    # get the im_info for different group, the im_info are the same for all images in one group
    I_frame = im_blob[0][0] # the I frame of the first group [frame_idx][frame_type_idx]
    im_info = np.array(
        [[I_frame.shape[0], I_frame.shape[1], im_scale]],
        dtype=np.float32)

    blobs = {'data': im_blob,
             'im_info': im_info}

    # ------------- gt boxes: (x1, y1, x2, y2, cls) --------------
    blobs['group_id'] = roidb[0]['group_id']
    all_gt_box = []
    for j in range(len(roidb)):
        if cfg.TRAIN.USE_ALL_GT:
            # Include all ground truth boxes
            gt_inds = np.where(roidb[j]['gt_classes'] != 0)[0]
        else:
            # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
            gt_inds = np.where((roidb[j]['gt_classes'] != 0) & np.all(roidb[j]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[j]['boxes'][gt_inds, :] * im_scale
        gt_boxes[:, 4] = roidb[j]['gt_classes'][gt_inds]

        all_gt_box.append(gt_boxes)

    blobs['boxes'] = all_gt_box
    return blobs


def _get_image_blob(roidb, target_size, group_size, im_type):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    return 2 list, each element in it is the data for one group (batch)
    """

    def load_frame_from_compressed_video(video_path, frame_id, frame_type, group_size=12):
        """
        This function load the frame from a compressed raw video.
        :param video_path: the path to mp4 raw video
        :param frame_id: int, starts from 1
        :param frame_type: int, 0 for I frame (also the image),
                    1 for motion vector, 2 for residual
        :param group_size: GOP, default is 12

        :return: ndarray, the loaded frame. For I fame and residual, it has
                    format BGR, for motion vector, the 0-th and 1-th channel
                     are x and y offsets respectively.
        """
        gop_idx = int((frame_id - 1) / group_size) # GOP starts from 0, while frame_id  here starts from 1.
        in_group_idx = int((frame_id - 1) % group_size) # the index in the group
        frame = coviar.load(video_path, gop_idx, in_group_idx, frame_type, True)

        return frame

    processed_ims = []
    for j in range(len(roidb)):

        frame_info = roidb[j]
        frame_path = frame_info['image'] # '/data0/liuqk/MOTChallenge/2DMOT2015/train/ETH-Bahnhof/img1/000229.jpg'
        frame_path_info = frame_path.split('/') # ['', 'data0', 'liuqk', 'MOTChallenge', '2DMOT2015', 'train', 'ETH-Bahnhof', 'img1', '000229.jpg']

        # get seq_path
        seq_path = '/'
        for info_idx in range(len(frame_path_info) - 2): # ['', 'data0', 'liuqk', 'MOTChallenge', '2DMOT2015', 'train', 'ETH-Bahnhof']
            seq_path = os.path.join(seq_path, frame_path_info[info_idx])

        # get the video path
        video_path = os.path.join(seq_path, frame_path_info[-3] + '.mp4')
        if not os.path.exists(video_path):
            raise RuntimeError(video_path + ' do not exists')

        frame_id = int(frame_path_info[-1][0:6])

        # load frame
        if j == 0: # I frame
            # im is a BGR image
            im = load_frame_from_compressed_video(video_path=video_path, frame_id=frame_id, frame_type=0, group_size=group_size)

            # check whether it is a gray image
            if len(im.shape) == 2:
                im = im[:, :, np.newaxis]
                im = np.concatenate((im, im, im), axis=2)

            if roidb[j]['flipped']:
                im = im[:, ::-1, :]

            im, scale_im = prep_im_for_blob(im=im,
                                            pixel_normal_scale=cfg.PIXEL_NORMAL_SCALE,
                                            pixel_means=cfg.PIXEL_MEANS,
                                            pixel_stds=cfg.PIXEL_STDS,
                                            target_size=target_size,
                                            channel=cfg.PIXEL_CHANNEL)
            mv, scale_mv = None, scale_im
            residual, scale_residual = None, scale_im

        elif j > 0:

            im = None

            mv = load_frame_from_compressed_video(video_path=video_path, frame_id=frame_id, frame_type=1, group_size=group_size)
            # residual can be treated as a BGR image
            residual = load_frame_from_compressed_video(video_path=video_path, frame_id=frame_id, frame_type=2, group_size=group_size)
            # check whether it is a gray image
            if len(residual.shape) == 2:
                residual = residual[:, :, np.newaxis]
                residual = np.concatenate((residual, residual, residual), axis=2)

            if roidb[j]['flipped']:
                residual = residual[:, ::-1, :]
                mv = mv[:, ::-1, :]
                # motion vector also needs to negative the x offsets
                mv[:, :, 0] = - mv[:, :, 0]

            mv, scale_mv = prep_mv_for_blob(im=mv,
                                            mv_normal_scale=cfg.MV_NORMAL_SCALE,
                                            mv_means=cfg.MV_MEANS,
                                            mv_stds=cfg.MV_STDS,
                                            target_size=target_size,
                                            channel=cfg.MV_CHANNEL)
            residual, scale_residual = prep_residual_for_blob(im=residual,
                                                              pixel_normal_scale=cfg.RESIDUAL_NORMAL_SCALE,
                                                              pixel_means=cfg.RESIDUAL_MEANS,
                                                              pixel_stds=cfg.RESIDUAL_STDS,
                                                              target_size=target_size,
                                                              channel=cfg.RESIDUAL_CHANNEL)

        # check the scales of im, mv and residual
        if scale_im != scale_mv or scale_im != scale_residual or scale_mv != scale_residual:
            raise RuntimeError(
                'the scales to resize I frame {}, motion vector {} and residual {} are not the same'.format(
                    scale_im, scale_mv, scale_residual))

        one_processed_im = [im, mv, residual]

        processed_ims.append(one_processed_im)

    # Create a blob to hold the input images
    # im_blob = group_ims_list_to_blob(processed_ims)


    return processed_ims, scale_im
