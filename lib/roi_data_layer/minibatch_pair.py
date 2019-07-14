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


def get_minibatch(roidb, num_classes, target_size, group_size, im_type=['im', 'residual', 'mv']):
    """Given a roidb, construct a minibatch sampled from it."""

    # Get the input image blob, formatted for caffe
    im_blob, im_scale = _get_image_blob(roidb, target_size, group_size, im_type=im_type)

    # get the im_info for different group, the im_info are the same for all images in one group
    I_frame = im_blob[0]
    im_info = np.array(
        [[I_frame.shape[0], I_frame.shape[1], im_scale]],
        dtype=np.float32)

    blobs = {'data': im_blob,
             'im_info': im_info
             }

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

        # gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        # gt_boxes[:, 0:4] = roidb[j]['boxes'][gt_inds, :] * im_scale
        # gt_boxes[:, 4] = roidb[j]['gt_classes'][gt_inds]

        gt_boxes = np.empty((len(gt_inds), 6), dtype=np.float32) # (x1, y1, x2, y2, cls, target_id)
        gt_boxes[:, 0:4] = roidb[j]['boxes'][gt_inds, :] * im_scale
        gt_boxes[:, 4] = roidb[j]['gt_classes'][gt_inds]
        gt_boxes[:, 5] = roidb[j]['target_id'][gt_inds]

        all_gt_box.append(gt_boxes)

    blobs['boxes'] = all_gt_box

    # # ------------ show some results ------------------------
    # from lib.model.utils.misc import show_compressed_frame
    # show_compressed_frame(np.array(blobs['data'][1][:, :, 0:3]+cfg.PIXEL_MEANS, dtype=np.uint8), 0)
    # show_compressed_frame(blobs['data'][1][:,:,3:5], 1)
    # show_compressed_frame(np.array(blobs['data'][1][:,:,5:8], dtype=np.uint8), 2)

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
        frame_path_info = frame_path.split('/')
        if 'motchallenge' in roidb[j]['dataset_name']:
            # ['', 'data0', 'liuqk', 'MOTChallenge', '2DMOT2015', 'train', 'ETH-Bahnhof', 'img1', '000229.jpg']

            # get seq_path
            seq_path = '/'
            for info_idx in range(len(frame_path_info) - 2):
                seq_path = os.path.join(seq_path, frame_path_info[info_idx])
            # get the video path
            video_path = os.path.join(seq_path, frame_path_info[-3] + '.mp4')
        elif 'citypersons' in roidb[j]['dataset_name']:
            # ['', 'data0', 'liuqk', 'Cityscapes', 'citysacpesdataset', 'leftImg8bit',
            #  'train', 'tubingen', 'tubingen_000082_000019_leftImg8bit', '000001.png']

            # get seq_path
            seq_path = '/'
            for info_idx in range(len(frame_path_info) - 1):
                seq_path = os.path.join(seq_path, frame_path_info[info_idx])
            # get the video path
            video_path = os.path.join(seq_path, frame_path_info[-2] + '.mp4')

        if not os.path.exists(video_path):
            raise RuntimeError(video_path + ' do not exists')
        frame_id = int(frame_path_info[-1][0:6])

        # load frame
        # im is a BGR image
        if 'im' in im_type:
            im = load_frame_from_compressed_video(video_path=video_path,
                                                  frame_id=frame_id, frame_type=0, group_size=group_size)
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
            im_shape = im.shape

        else:
            im = None

        if 'mv' in im_type:
            mv = load_frame_from_compressed_video(video_path=video_path, frame_id=frame_id, frame_type=1, group_size=group_size)

            if roidb[j]['flipped']:
                mv = mv[:, ::-1, :]
                mv[:, :, 0] = - mv[:, :, 0]

            mv, scale_im = prep_mv_for_blob(im=mv,
                                            mv_normal_scale=cfg.MV_NORMAL_SCALE,
                                            mv_means=cfg.MV_MEANS,
                                            mv_stds=cfg.MV_STDS,
                                            target_size=target_size,
                                            channel=cfg.MV_CHANNEL)

            im_shape = mv.shape

        else:
            mv = None

        if 'residual' in im_type:
            residual = load_frame_from_compressed_video(video_path=video_path, frame_id=frame_id, frame_type=2, group_size=group_size)
            # check whether it is a gray image
            if len(residual.shape) == 2:
                residual = residual[:, :, np.newaxis]
                residual = np.concatenate((residual, residual, residual), axis=2)

            if roidb[j]['flipped']:
                residual = residual[:, ::-1, :]

            residual, scale_im = prep_residual_for_blob(im=residual,
                                                        pixel_normal_scale=cfg.RESIDUAL_NORMAL_SCALE,
                                                        pixel_means=cfg.RESIDUAL_MEANS,
                                                        pixel_stds=cfg.RESIDUAL_STDS,
                                                        target_size=target_size,
                                                        channel=cfg.RESIDUAL_CHANNEL)

            im_shape = residual.shape
        else:
            residual = None

        one_processed_im = np.zeros((im_shape[0], im_shape[1], 3+2+3))
        if im is not None:
            one_processed_im[:,:,0:3] = im
        if mv is not None:
            one_processed_im[:,:,3:5] = mv
        if residual is not None:
            one_processed_im[:,:,5:8] = residual

        # # ------------ show some results ------------------------
        # from lib.model.utils.misc import show_compressed_frame
        # show_compressed_frame(np.array(one_processed_im[:, :, 0:3]+cfg.PIXEL_MEANS, dtype=np.uint8), 0)
        # show_compressed_frame(one_processed_im[:,:,3:5], 1)
        # show_compressed_frame(np.array(one_processed_im[:,:,5:8], dtype=np.uint8), 2)

        # # ------------ show some results ------------------------
        # from lib.model.utils.misc import show_compressed_frame
        # show_compressed_frame(mv, 1)
        processed_ims.append(one_processed_im)

    return processed_ims, scale_im
