from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import PIL
import pickle
from lib.datasets.imdb_pair import imdb
import pandas
from lib.datasets.tools.misc import filter_mot_gt_boxes, jitter_a_box
from lib.datasets.imdb_pair import ROOT_DIR

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from lib.model.utils.config import cfg
import coviar

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete


class motchallenge(imdb):
    def __init__(self, phase, name='motchallenge_tracking_pair', mot_database_path=None, train_ratio=1,
                 jitter=True, jitter_iou=0.8):
        imdb.__init__(self, name)

        # the group size in the raw mpeg4 video
        self._group_size = 12

        # Since there is no validation data, we divide origin train dataset into
        # train data and validation data. The first TRAIN_RATIO GOPs of a sequence
        # are used to train, and the last 1 - TRAIN_RATIO GOPs are used to validation
        # the  ratio of train data to origin train data
        self._train_ratio = train_ratio

        self._phase = phase # phase can be 'train' or 'val'

        self.jitter = jitter # whether to jitter the box in the first frame of one pair
        self.jitter_iou = jitter_iou # the iou threshold to jitter the gt box

        # The sequences in MOT16 and MOT17 are the same, while the sequences in
        # 2DMOT2015 are not all the same with those in MOT17. To handle this, we
        # filter 2DMOT2015 dataset, i.e. those sequences that are contained in
        # MOT17 will not be included in 2DMOT2015
        # self._2DMOT2015 = ['ETH-Bahnhof', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte']
        # self._MOT17 = ['MOT17-04', 'MOT17-13', 'MOT17-11', 'MOT17-10', 'MOT17-09', 'MOT17-05', 'MOT17-02']
        self._2DMOT2015 = {'train': ['ETH-Bahnhof', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus',
                                     'TUD-Stadtmitte'],
                           'test': ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli',
                                    'ETH-Linthescher',
                                    'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1'],
                           'val': []
                           }
        self._MOT17 = {'train': ['MOT17-04', 'MOT17-11', 'MOT17-05', 'MOT17-13', 'MOT17-02'],
                       'test': ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14'],
                       'val': ['MOT17-10', 'MOT17-09']
                       }
        self._sequences = {'test': self._MOT17['train'] + self._2DMOT2015['test'],
                           'train': self._MOT17['train'] + self._2DMOT2015['train'],
                           'val': self._MOT17['val'] + self._2DMOT2015['val']}

        # self._2DMOT2015 = ['KITTI-13']
        # self._MOT17 = ['MOT17-13']
        self._data_path = mot_database_path if mot_database_path is not None else '/home/liuqk/Dataset/MOT'

        # there are only 2 classes in MOTChallenge
        self._classes = ('__background__', 'person')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'

        # we store the frame index for all frames in MOT17 + 2DMOT2015
        # one frame has the index such as: 2DMOT2015/train/AUD-Campus/img1/000001.jpg
        self._image_index = self._load_frame_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # PASCAL specific config options
        self.config = {'cleanup': True}

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def _load_frame_index(self):
        """
        The database are sequences, the frames in different sequences may have
        the same frame id. In order to distinguish them from each other, we store
        the absolute path for all frames
        :return: a list, each element in it is a string, i.e. the asbolute path of
                all frames in 2DMOT2015 and MOT17.
        """
        train_ratio = '{:.3f}'.format(self._train_ratio)
        cache_file_index = os.path.join(self.cache_path, self.name + '_' + train_ratio + '_image_index.pkl')
        if os.path.exists(cache_file_index):
            with open(cache_file_index, 'rb') as fid:
                pair_of_images = pickle.load(fid)
            print('{} pair of image index loaded from {}'.format(self.name, cache_file_index))

        else:
            if self._phase in ['train', 'training']:
                sequences = self._sequences['train']
                MOT15 = self._2DMOT2015['train']
                MOT17 = self._MOT17['train']
            elif self._phase in ['val', 'validation']:
                sequences = self._sequences['val']
                MOT15 = self._2DMOT2015['val']
                MOT17 = self._MOT17['val']
            elif self._phase in ['test', 'testing']:
                sequences = self._sequences['test']
                MOT15 = self._2DMOT2015['test']
                MOT17 = self._MOT17['test']
                self._train_ratio = 1 # all data used to test

            group_of_images = []
            group_count = -1
            for seq in sequences:
                dataset = None
                if seq in MOT15:
                    dataset = '2DMOT2015'
                    print(self._phase + ': loading image index of ' + dataset + ' ' + seq + '...')
                elif seq in MOT17:
                    dataset = 'MOT17'
                    print(self._phase + ': loading image index of ' + dataset + ' ' + seq + '...')
                    # MOT17 provides the detections of 3 detector, and the names of sequences
                    # are modified with the name of detector, but the gt files are the same
                    seq = seq + '-DPM'

                # find the length (number of frames) of this seq
                test_or_train = 'test' if self._phase in ['test', 'testing'] else 'train'
                images_path = os.path.join(self._data_path, dataset, test_or_train, seq, 'img1')
                images_name = os.listdir(images_path)
                images_name.sort()

                # pick up the train or val images
                num_gops = int(len(images_name) / self._group_size)
                num_train_gops = int(num_gops * self._train_ratio)

                first_im_id, last_im_id = 1, len(images_name) - 1
                if self._train_ratio != 1:
                    if self._phase in ['train', 'training']:
                        first_im_id = 1
                        last_im_id = int(num_train_gops * self._group_size)
                    elif self._phase in ['val', 'validation']:
                        first_im_id = int(num_train_gops * self._group_size) + 1
                        last_im_id = len(images_name)
                    else:
                        raise RuntimeError('Expected pahse is validation and train, but get unrecognized pahse {}'.format(self._phase))

                # we find the frame id in the gt file since some frames have no gt box
                gt_file_path = os.path.join(self._data_path, dataset, 'train', seq, 'gt', 'gt.txt')
                #gt_boxes = np.loadtxt(gt_file_path, delimiter=',')
                gt_boxes = pandas.read_csv(gt_file_path).values

                # filter out other classes for MOT17
                if dataset == 'MOT17':
                    gt_boxes = filter_mot_gt_boxes(gt_boxes=gt_boxes, vis_threshold=cfg.TRAIN.VISIBILITY_THRESHOLD,
                                                   ambiguous_class_id=cfg.TRAIN.AMBIGUOUS_CLASS_ID)

                # pick the train or val set
                index = gt_boxes[:, 0] >= first_im_id
                gt_boxes = gt_boxes[index]
                index = gt_boxes[:, 0] <= last_im_id
                gt_boxes = gt_boxes[index]

                # find the image id based on the frame id in gt file
                frame_id = np.unique(gt_boxes[:, 0])
                frame_id = np.asarray(frame_id, dtype=np.int32)
                frame_id = np.sort(frame_id, axis=0)

                # make the frame id start from 1 or 13 or 25 ...
                i = 1
                while (frame_id[0] - 1) % self._group_size != 0:
                    idx = frame_id > self._group_size * i
                    frame_id = frame_id[idx]
                    i += 1

                for i in range(len(frame_id)):
                    one_id = frame_id[i]

                    # the last frame of this video was removed, because we found
                    # that the motion vector and residuals of the last frame are
                    # all zeros
                    if one_id == len(images_name):
                        continue

                    one_index = dataset + '/' + 'train' + '/' + seq + '/' + 'img1' + '/' + str(one_id).zfill(6) + self._image_ext

                    if (one_id - 1) % self._group_size == 0:
                        group_of_images.append([])
                        group_count += 1
                        group_of_images[group_count].append(one_index)
                        i_frame_id = one_id
                    elif one_id > i_frame_id and one_id < i_frame_id + self._group_size:
                        group_of_images[group_count].append(one_index)
                    else:
                        pass

            # check group of images
            print('check group of images...')
            for g_idx in range(len(group_of_images)):
                index_1 = group_of_images[g_idx][0]
                index_ls = index_1.split('/')
                frame_id_1 = int(index_ls[4][0:6])

                for idx in range(1, len(group_of_images[g_idx])):
                    index_2 = group_of_images[g_idx][idx]
                    index_ls = index_2.split('/')
                    frame_id_2 = int(index_ls[4][0:6])

                    if (frame_id_1 - 1) % self._group_size == 0:
                        if frame_id_1 < frame_id_2 and frame_id_2 < frame_id_1 + 12:
                            pass
                        else:
                            raise RuntimeError('Found invalid pair of image pair:\n index 1: {}\n index 2: {}'.
                                               format(index_1, index_2))
                    else:
                        raise RuntimeError('Found invalid pair of image pair:\n index 1: {}\n index 2: {}'.
                                           format(index_1, index_2))

            # load gt of group of images, so that we check whether there are shared targets in both frame
            print('load gt of group of images...')
            train_ratio = '{:.3f}'.format(self._train_ratio)
            cache_file_group_of_images = os.path.join(self.cache_path, self.name + '_' + train_ratio + '_group_of_images_gt_roidb.pkl')
            if os.path.exists(cache_file_group_of_images):
                with open(cache_file_group_of_images, 'rb') as fid:
                    group_of_images_gt = pickle.load(fid)
            else:
                group_of_images_gt = []
                for idx in range(len(group_of_images)):
                    one_group_of_images_gt = [self._load_mot_annotation(index)
                                        for index in group_of_images[idx]]
                    group_of_images_gt.append(one_group_of_images_gt)
                with open(cache_file_group_of_images, 'wb') as fid:
                    pickle.dump(group_of_images_gt, fid, pickle.HIGHEST_PROTOCOL)

            # change group index into pair index
            # there are two kind of frame pair.
            # 1) choose I fame as the first one frame, and choose a p frame from the same group
            # 2) choose 2 successive frame from the same group.

            pair_of_images = []
            # we first generate the first kind of pairs
            print('change group index into pair index...')
            for g_idx in range(len(group_of_images)):
                for idx in range(1, len(group_of_images[g_idx])):
                    one_pair = [group_of_images[g_idx][0], group_of_images[g_idx][idx]]
                    target_id_1 = group_of_images_gt[g_idx][0]['target_id']
                    target_id_2 = group_of_images_gt[g_idx][idx]['target_id']
                    target_id_share = []
                    for id_idx in range(target_id_1.shape[0]):
                        if target_id_1[id_idx] in target_id_2:
                            target_id_share.append(target_id_1[id_idx])
                            break
                    if len(target_id_share) > 0:
                        pair_of_images.append(one_pair)

            # then we generate the second kind of pairs
            for g_idx in range(len(group_of_images)):
                for idx in range(1, len(group_of_images[g_idx]) - 1):
                    # check whether this two frames share some targets
                    target_id_1 = group_of_images_gt[g_idx][idx]['target_id']
                    target_id_2 = group_of_images_gt[g_idx][idx + 1]['target_id']
                    target_id_share = []
                    for id_idx in range(target_id_1.shape[0]):
                        if target_id_1[id_idx] in target_id_2:
                            target_id_share.append(target_id_1[id_idx])
                            break
                    if len(target_id_share) > 0:
                        index_1 = group_of_images[g_idx][idx]
                        index_2 = group_of_images[g_idx][idx + 1]

                        index_ls = index_1.split('/')
                        frame_id_1 = int(index_ls[4][0:6])

                        index_ls = index_2.split('/')
                        frame_id_2 = int(index_ls[4][0:6])

                        if frame_id_1 + 1 == frame_id_2:
                            one_pair = [index_1, index_2]
                            pair_of_images.append(one_pair)
                        elif frame_id_2 + 1 == frame_id_1:
                            one_pair = [index_2, index_1]
                            pair_of_images.append(one_pair)

            # check each pair
            print('check each pair...')
            for idx in range(len(pair_of_images)):
                if len(pair_of_images[idx]) != 2:
                    raise RuntimeError('Found one pair with the length {}, but expected is 2'.format(len(pair_of_images[idx])))
                index_1 = pair_of_images[idx][0]
                index_2 = pair_of_images[idx][1]

                index_ls = index_1.split('/')
                frame_id_1 = int(index_ls[4][0:6])

                index_ls = index_2.split('/')
                frame_id_2 = int(index_ls[4][0:6])

                if (frame_id_1-1) % self._group_size == 0:
                    if frame_id_1 < frame_id_2 and frame_id_2 < frame_id_1 + self._group_size:
                        pass
                    elif frame_id_1 + 1 == frame_id_2:
                        pass
                    else:
                        raise RuntimeError('Found invalid pair of image pair:\n index 1: {}\n index 2: {}'.
                                           format(index_1, index_2))

            # print('filtering the pairs...')
            # pair_of_images = self.filter_pair_of_images(pair_of_images)

            with open(cache_file_index, 'wb') as fid:
                pickle.dump(pair_of_images, fid, pickle.HIGHEST_PROTOCOL)

        return pair_of_images

    def filter_pair_of_images(self, pair_of_images):
        """
        This function filter those bad pairs.
        :param pair_of_images: list, each element in it is also a list (with length 2, each is a index)
        :return:
        """

        def load_frame_from_compressed_video(video_path, frame_id, frame_type, accumulated, group_size=12):
            """
            This function load the frame from a compressed raw video.
            :param video_path: the path to mp4 raw video
            :param frame_id: int, starts from 1
            :param frame_type: int, 0 for I frame (also the image),
                        1 for motion vector, 2 for residual
            :param accumulated: bool, determin whether to loaded accumulated mv or residual
            :param group_size: GOP, default is 12

            :return: ndarray, the loaded frame. For I fame and residual, it has
                        format BGR, for motion vector, the 0-th and 1-th channel
                         are x and y offsets respectively.
            """
            gop_idx = int((frame_id - 1) / group_size)  # GOP starts from 0, while frame_id  here starts from 1.
            in_group_idx = int((frame_id - 1) % group_size)  # the index in the group
            frame_load = coviar.load(video_path, gop_idx, in_group_idx, frame_type, accumulated)
            return frame_load

        pair_of_images_filter = []
        for pair_idx in range(len(pair_of_images)):

            print('Filtering {}/{}'.format(pair_idx, len(pair_of_images)))

            one_pair = pair_of_images[pair_idx]
            # 'MOT17/train/MOT17-04-DPM/img1/000937.jpg'
            index_1 = one_pair[0]
            index_2 = one_pair[1]

            index_1_ls = index_1.split('/')
            index_2_ls = index_2.split('/')

            frame_id_1 = int(index_1_ls[4][0:6])
            frame_id_2 = int(index_2_ls[4][0:6])

            if (frame_id_1 - 1) % 12 == 0:
                accumulated = True
            elif (frame_id_1 - 1) % 12 != 0 and frame_id_1 + 1 == frame_id_2:
                accumulated = False
            else:
                raise RuntimeError(
                    'Invalid tracking pair!\n image1: {}\n image2: {}'.format(index_1, index_2))

            video_path = os.path.join(self._data_path, index_1_ls[0], index_1_ls[1], index_1_ls[2], index_1_ls[2] + '.mp4')

            mv_1 = load_frame_from_compressed_video(video_path, frame_id_1, 1, accumulated)
            resdual_1 = load_frame_from_compressed_video(video_path, frame_id_1, 2, accumulated)

            mv_2 = load_frame_from_compressed_video(video_path, frame_id_2, 1, accumulated)
            resdual_2 = load_frame_from_compressed_video(video_path, frame_id_2, 2, accumulated)

            if (frame_id_1 - 1) % self._group_size == 0: # the  mv and residual of I frame are all zeros
                if np.sum(np.fabs(mv_2)) + np.sum(np.fabs(resdual_2)) > 0:
                    pair_of_images_filter.append(one_pair)

            else:
                if np.sum(np.fabs(mv_1)) + np.sum(np.fabs(resdual_1)) > 0 and np.sum(np.fabs(mv_2)) + np.sum(np.fabs(resdual_2)) > 0:
                    pair_of_images_filter.append(one_pair)

        print('Before filtering, there are {} pairs, after filtering, there are {} paires.'.
              format(len(pair_of_images), len(pair_of_images_filter)))

        return pair_of_images_filter


    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        the index has the format:  2DMOT2015/train/AUD-Campus/img1/000001.jpg
        """
        # in linux os, we can directly return os.path.join(self._data_path, index)
        index_ls = index.split('/')
        image_path = self._data_path
        for i in range(len(index_ls)):
            image_path = os.path.join(image_path, index_ls[i])

        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def image_at_group_id(self, i):
        """
        return the group id of the frames
        :param i:
        :return:
        """
        return i

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        train_ratio = '{:.3f}'.format(self._train_ratio)
        cache_file_gt = os.path.join(self.cache_path, self.name + '_' + train_ratio + '_gt_roidb.pkl')
        if os.path.exists(cache_file_gt):
            with open(cache_file_gt, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file_gt))
            #return roidb
        else:

            print('load gt boxes for all frames...')
            roidb = []
            for i in range(len(self._image_index)):
                print('load gt of {}/{} frame pair'.format(i+1, len(self._image_index)))
                one_group_gt = [self._load_mot_annotation(index)
                                for index in self._image_index[i]]
                roidb.append(one_group_gt)
            with open(cache_file_gt, 'wb') as fid:
                pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file_gt))

        if self.jitter:
            cache_file_jitter = os.path.join(self.cache_path,
                                             self.name + '_jitter_' + train_ratio + '_gt_roidb.pkl')
            if os.path.exists(cache_file_jitter):
                with open(cache_file_jitter, 'rb') as fid:
                    roidb = pickle.load(fid)
                print('{} gt roidb loaded from {}'.format(self.name, cache_file_jitter))
                # return roidb
            else:
                for idx in range(len(roidb)):
                    print('Jitter the box of {}/{} pair...'.format(idx + 1, len(roidb)))

                    one_pair_ann = roidb[idx]
                    ann_1 = one_pair_ann[0]

                    im_path = os.path.join(self._data_path, ann_1['index'])
                    im_size = PIL.Image.open(im_path).size
                    im_w, im_h = im_size[0], im_size[1]

                    gt_bbox = ann_1['boxes'] # [x1, y1, x2, y2]

                    gt_bbox_xywh = gt_bbox.copy()
                    gt_bbox_xywh[:, 2:4] = gt_bbox_xywh[:, 2:4] - gt_bbox_xywh[:, 0:2] + 1
                    gt_bbox_xywh_jitter = gt_bbox_xywh.copy()

                    for box_idx in range(gt_bbox_xywh.shape[0]):
                        one_box = gt_bbox_xywh[box_idx, :]
                        one_box_jitter = jitter_a_box(one_box=one_box, iou_thr=self.jitter_iou, up_or_low='up')
                        gt_bbox_xywh_jitter[box_idx, :] = one_box_jitter

                    gt_bbox_jitter = gt_bbox_xywh_jitter.copy()
                    gt_bbox_jitter[:, 2:4] = gt_bbox_jitter[:, 2:4] + gt_bbox_jitter[:, 0:2] - 1

                    x1 = gt_bbox_xywh_jitter[:, 0]
                    y1 = gt_bbox_xywh_jitter[:, 1]
                    w = gt_bbox_xywh_jitter[:, 2]
                    h = gt_bbox_xywh_jitter[:, 3]
                    x2 = x1 + w - 1
                    y2 = y1 + h - 1
                    refined_bbox = gt_bbox_xywh_jitter.copy()
                    refined_bbox[:, 0] = x1.clip(0, im_w - 1)
                    refined_bbox[:, 1] = y1.clip(0, im_h - 1)
                    refined_bbox[:, 2] = x2.clip(0, im_w - 1)
                    refined_bbox[:, 3] = y2.clip(0, im_h - 1)

                    ann_1['boxes'] = refined_bbox

                    roidb[idx][0] = ann_1

                with open(cache_file_jitter, 'wb') as fid:
                    pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
                print('wrote gt roidb to {}'.format(cache_file_jitter))

        return roidb

    def _load_mot_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        the index has the format:  2DMOT2015/train/AUD-Campus/img1/000001.jpg
        """
        #print('Load gt of ' + index)
        index_ls = index.split('/')

        frame_id = int(index_ls[4][0:6])
        frame_type = 'i_frame' if (frame_id-1) % self._group_size == 0 else 'p_frame'
        dataset = index_ls[0]
        seq = index_ls[2]
        # get the image size
        frame_name = str(frame_id).zfill(6) + self._image_ext
        fram_path = os.path.join(self._data_path, dataset, 'train', seq, 'img1', frame_name)
        im_size = PIL.Image.open(fram_path).size
        im_w, im_h = im_size[0], im_size[1]

        # load gt box
        gt_file_path = os.path.join(self._data_path, dataset, 'train', seq, 'gt', 'gt.txt')
        gt = pandas.read_csv(gt_file_path).values
        #gt = np.loadtxt(gt_file_path, delimiter=',')

        # filter out other classes for MOT17
        if dataset == 'MOT17':
            gt = filter_mot_gt_boxes(gt_boxes=gt, vis_threshold=cfg.TRAIN.VISIBILITY_THRESHOLD,
                                     ambiguous_class_id=cfg.TRAIN.AMBIGUOUS_CLASS_ID)

        idx = gt[:, 0] == frame_id
        gt = gt[idx]

        target_id = gt[:, 1]

        bbox = gt[:, 2:6]  # x1, y1, w, h
        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        w = bbox[:, 2]
        h = bbox[:, 3]
        x2 = x1 + w - 1
        y2 = y1 + h - 1
        refined_bbox = bbox.copy()
        refined_bbox[:, 0] = x1.clip(0, im_w - 1)
        refined_bbox[:, 1] = y1.clip(0, im_h - 1)
        refined_bbox[:, 2] = x2.clip(0, im_w - 1)
        refined_bbox[:, 3] = y2.clip(0, im_h - 1)

        # check the boxes whether all in the frame (boxes maybe out of the image view)
        if min(refined_bbox[:, 0]) < 0 or min(refined_bbox[:, 1]) < 0 or \
                min(refined_bbox[:, 2]) >= im_w or min(refined_bbox[:, 3]) >= im_h:
            raise RuntimeError('Box out the image view')

        if ((refined_bbox[:, 0] <= refined_bbox[:, 2]).all() == False) or ((refined_bbox[:, 1] <= refined_bbox[:, 3]).all() == False):
            raise RuntimeError('Find invalid boxes: x1 >= x2 or y1 >= y2')

        gt_classes = np.ones((bbox.shape[0]), dtype=np.int32)
        if dataset == 'MOT17':
            idx = gt[:, 7] == 8  # the 'Distractor' class
            if cfg.TRAIN.AMBIGUOUS_CLASS_ID is None:
                if idx.sum() > 0:
                    raise ValueError('Find ambiguous gt box, but config.TRAIN.AMBIGUOUS_CLASS_ID is None! ')
            else:
                gt_classes[idx] = cfg.TRAIN.AMBIGUOUS_CLASS_ID

        overlaps = np.zeros((bbox.shape[0], self.num_classes), dtype=np.float32)
        overlaps[gt_classes == 1, 1] = 1.0  # set person class boxes to the corresponding index is 1
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': refined_bbox, # [x1, y1, x2, y2]
                'gt_classes': gt_classes,
                'target_id': target_id,
                'gt_overlaps': overlaps,
                'flipped': False,
                'index': index,
                'frame_type': frame_type,
                'dataset_name': self.name}

if __name__ == '__main__':
    d = motchallenge(name='motchallenge')
    res = d.roidb
    from IPython import embed;

    embed()
