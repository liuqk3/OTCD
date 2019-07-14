
import pandas as pd
import numpy as np
import torch
import os
from lib.datasets.tools.misc import filter_mot_gt_boxes, get_one_tracklet, get_one_patch_pair
from lib.model.utils.config import cfg
import random
import matplotlib.pyplot as plt
from lib.model.roi_align.roi_align.roi_align import RoIAlign
from torch.autograd import Variable

class motchallenge_tracklets(object):

    def __init__(self, phase, name='motchallenge_tracklets', mot_dataset_path=None, crop_h=7, crop_w=7, cache_dir=None):
        """
        This class used to generate tracklets, so we can train the sbc_net to perform online tracking.
        :param phase: string, 'train', 'val'
        :param name: the name of this class
        :param mot_dataset_path: the root path to motchallenge dataset
        :param crop_h: the height of croped feature
        :param crop_w: the width of croped feature
        :param cache_dir: str, the path to cache the data
        """

        self._phase = phase
        self.name = name
        self.num_tracklets = None
        self.negative_positive_ratio = None
        self.max_length = None
        self.min_length = None
        self.interval = None
        self.iou_thr = None
        self.cache_dir = cache_dir if cache_dir is not None else os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.crop_h = crop_h
        self.crop_w = crop_w
        self.roi_align = RoIAlign(crop_width=crop_w, crop_height=crop_h, transform_fpcoor=True)

        self._data_path = '/home/liuqk/Dataset/MOT' if mot_dataset_path is None else motchallenge_tracklets
        if not os.path.exists(self._data_path):
            raise ValueError('Dataset path: {} does not exists!'.format(self._data_path))

        # The sequences in MOT16 and MOT17 are the same, while the sequences in
        # 2DMOT2015 are not all the same with those in MOT17. To handle this, we
        # filter 2DMOT2015 dataset, i.e. those sequences that are contained in
        # MOT17 will not be included in 2DMOT2015
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

        self.tracklets = None #self.generate_tracklets()

    def generate_tracklets(self, num_tracklets=1e4, negative_positive_ratio=3, max_length=12, min_length=2, interval=1,
                           jitter=False, iou_thr=0.7, shuffle=False, regenerate=False, tracklet_or_pair=None):
        """
        This function used to generate tracklets.
        :param num_tracklets: the number of tracklets.
        :param negative_positive_ratio: the ratio between the number of negative and positive tracklets.
        :param max_length: scalar, the max number of detections (frames) boxes in one tracklet.
        :param min_length: scalar, the min number of detections (frames) boxes in one tracklets.
        :param interval: scalar, the boxes in one tracklet is sampled from every 'interval' frames.
        :param jitter: bool, whether to jitter tracklets. If False, the boxes in tracklets are the gt boxes,
                    if True, the boxes are jittered with Gaussian distribution.
        :param iou_thr: the boxes jittered from the origin boxes will be kept if iou > iou_thr
        :param shuffle: bool, whether to shuffle the boxes in one tracklet
        :param regenerate: bool, whether to regenerate the tracklets. If True, we will generate new tracklets,
                    otherwise, we will load the tracklets from the cache
        :param tracklet_or_pair: string, 'tracklet' or 'pair'. If tracklet, we generate tracklets with the assigned
                    length, otherwise, we generate a pair of path
        :return: a list, with length of num_tracklts. Each element in it is an array, contain the information of
                    box in one tracklet. Each row in one tracklet is:
                    for MOT17: [frame_id, target_id, x1, y1, x2, y2, consideration, category_id, visibility]
                    for MOT15: [frame_id, target_id, x1, y1, x2, y2, consideration, -1, -1, -1]
        """
        self.num_tracklets = int(num_tracklets)
        self.negative_positive_ratio = negative_positive_ratio
        self.max_length = int(max_length)
        self.min_length = int(min_length)
        self.interval = int(interval)
        self.iou_thr = iou_thr

        tracklets_cache_name = self.name + '_' + self._phase + '_' + str(self.num_tracklets) + '_' +\
                               str(self.negative_positive_ratio) + '_' + str(self.max_length) + '_' +\
                               str(self.min_length) + '_' + str(self.interval) + '_' + tracklet_or_pair + '.pkl'
        tracklets_cache_path = os.path.join(self.cache_dir, tracklets_cache_name)
        exists = False

        if not regenerate: # load the tracklets from the cache
            if os.path.exists(tracklets_cache_path):
                exists = True
                self.tracklets = torch.load(tracklets_cache_path)
            else:
                exists = False

        if not exists: # generate the tracklets
            if self._phase in ['train', 'training']:
                sequences = self._sequences['train']
                MOT15 = self._2DMOT2015['train']
                MOT17 = self._MOT17['train']
                train_or_test = 'train'
            elif self._phase in ['val', 'validation']:
                sequences = self._sequences['val']
                MOT15 = self._2DMOT2015['val']
                MOT17 = self._MOT17['val']
                train_or_test = 'train'
            elif self._phase in ['test', 'testing']:
                sequences = self._sequences['test']
                MOT15 = self._2DMOT2015['test']
                MOT17 = self._MOT17['test']
                train_or_test = 'test'

            num_positive = self.num_tracklets // (1 + self.negative_positive_ratio)
            num_negative = self.num_tracklets - num_positive

            tracklets = []
            # generate tracklets
            for i in range(int(num_tracklets)):
                print('generate tracklets: {}/{}'.format(i+1, num_tracklets))
                # for each tracklet, we random choose a seq and the length
                seq = sequences[np.random.choice(range(len(sequences)))]
                length = np.random.choice(range(self.min_length, self.max_length + 1))
                if length < min_length or length > max_length:
                    raise RuntimeError('The length {} of tracket is out of range [{}, {}]'.
                                       format(length, min_length, max_length))

                if seq in MOT15:
                    data_year = '2DMOT2015'
                elif seq in MOT17:
                    data_year = 'MOT17'
                    seq = seq + '-DPM'

                seq_path = os.path.join(self._data_path, data_year, train_or_test, seq)
                frame_names = os.listdir(os.path.join(seq_path, 'img1'))
                frame_names.sort()
                num_frames = len(frame_names)

                one_im_path = os.path.join(seq_path, 'img1', '000001.jpg')
                im_shape = np.array(plt.imread(one_im_path).shape) #[h, w, c]

                if not os.path.exists(seq_path):
                    raise RuntimeError('Path does not exists: {}'.format(seq_path))

                gt_file_path = os.path.join(seq_path, 'gt', 'gt.txt')
                gt_boxes = pd.read_csv(gt_file_path).values
                if data_year == 'MOT17':
                    # There are 12 classes in MOT16 and MOT17.
                    # we set the class_id 1, 2, 7 as positive, 8 as ambiguous, and the other as negative.
                    # So we only need to preserve the 1, 2, 7, 8 classes
                    gt_boxes = filter_mot_gt_boxes(gt_boxes=gt_boxes, vis_threshold=cfg.TRAIN.VISIBILITY_THRESHOLD,
                                                   ambiguous_class_id=cfg.TRAIN.AMBIGUOUS_CLASS_ID)

                # we need to filter out the last frame, because we do not have the residual and mv for it
                index = gt_boxes[:, 0] < num_frames
                gt_boxes = gt_boxes[index]

                if i < num_positive:
                    if tracklet_or_pair == 'tracklet':
                        a_tracklet = get_one_tracklet(gt_boxes=gt_boxes, length=length, type='positive', target_id=None,
                                                        interval=self.interval,  shuffle=shuffle, jitter=jitter,
                                                        iou_thr=self.iou_thr, im_shape=im_shape)
                    elif tracklet_or_pair == 'pair':
                        a_tracklet = get_one_patch_pair(gt_boxes=gt_boxes, length=length, type='positive', target_id=None,
                                                       jitter=jitter, iou_thr=self.iou_thr, im_shape=im_shape)
                    label = 1

                else:
                    if tracklet_or_pair == 'tracklet':
                        a_tracklet = get_one_tracklet(gt_boxes=gt_boxes, length=length, type='negative', target_id=None,
                                                        interval=self.interval,  shuffle=shuffle, jitter=True,
                                                        iou_thr=self.iou_thr, im_shape=im_shape)
                    elif tracklet_or_pair == 'pair':
                        a_tracklet = get_one_patch_pair(gt_boxes=gt_boxes, length=length, type='negative', target_id=None,
                                                       jitter=jitter, iou_thr=self.iou_thr, im_shape=im_shape)
                    label = 0

                one_tracklet = {'tracklet': a_tracklet,
                                'label': label,
                                'seq': seq,
                                'dataset': data_year,
                                'phase': train_or_test}

                tracklets.append(one_tracklet)

            random.shuffle(tracklets)
            torch.save(tracklets, tracklets_cache_path)
            self.tracklets = tracklets
            #return tracklets

    def get_rnn_input(self, i, feature_type='origin'):
        """
        This function prepare the training data for the i-th tracklet
        in self.trackltes
        :param i: the index of tracklets in self.tracklets. Starts from 0.
        :param feature_type: string, 'origin', 'warp'. Denote the origin feature or
                warped feature for this frame.
        :return: crop_f: tensor, the croped feature for the boxes in the tracklet, has
                    the size [length, bs, ,c, crop_h, crop_w]. Noted bs here is set to 1.
        """
        # for MOT17: [frame_id, target_id, x1, y1, x2, y2, consideration, category_id, visibility]
        # for MOT15: [frame_id, target_id, x1, y1, x2, y2, consideration, -1, -1, -1]

        one_tracklet = self.tracklets[i] # array
        data_year = one_tracklet['dataset']
        phase = one_tracklet['phase']
        seq = one_tracklet['seq']
        seq_path = os.path.join(self._data_path, data_year, phase, seq)

        tracklet_box = one_tracklet['tracklet']
        tracklet_length = tracklet_box.shape[0]
        label = one_tracklet['label']
        crop_f = None
        for i in range(tracklet_length):
            one_d = tracklet_box[i]
            frame_id = int(one_d[0])
            bbox = torch.FloatTensor(one_d[2:6]).unsqueeze(dim=0) # [1, 4]

            feature_path = os.path.join(seq_path, 'feature', str(frame_id).zfill(6)+'.pkl')
            if not os.path.exists(feature_path):
                raise RuntimeError('feature path: {} does not exists! '.format(feature_path))

            feature = torch.load(feature_path)

            # it should be noted that image are resized to [resize_h, resized_w]. And min(resize_h, resize_w)>=600
            # max(resize_h, resize_w) <= 1000
            feature_map = feature[feature_type] # [1, c, f_h, f_w], 1 is the batch index
            if feature_map is None: # the 'warp' feature map is Non for I frame
                feature_map = feature['origin']
            feature_map = feature_map.contiguous()

            im_info = feature['im_info'] # [1, 4], [resized_h, resized_w, scale] scale = resize_h / im_h = resize_w / im_w

            # map the bbox to the feature map
            f_h, f_w = feature_map.size()[2], feature_map.size()[3]
            h_scale = f_h / im_info[0, 0] * im_info[0, 2]
            w_scale = f_w / im_info[0, 1] * im_info[0, 2]
            bbox = bbox * torch.FloatTensor([w_scale, h_scale, w_scale, h_scale])

            # crop the feature for this bbox
            bbox_index = torch.zeros(1).int()

            feature_map = Variable(feature_map)
            bbox = Variable(bbox)
            bbox_index = Variable(bbox_index)
            if i == 0:
                crop_f = self.roi_align(feature_map, bbox, bbox_index).data.contiguous() # [1, c, crop_h, crop_w]
                crop_f = crop_f.unsqueeze(dim=0) # [1, 1, c, crop_h, crop_w]
            else:
                one_crop_f = self.roi_align(feature_map, bbox, bbox_index).data.contiguous()
                one_crop_f = one_crop_f.unsqueeze(dim=0)  # [1, 1, c, crop_h, crop_w]
                crop_f = torch.cat((crop_f, one_crop_f), dim=0)

        num_crops = crop_f.size()[0]
        if num_crops != tracklet_length:
            raise RuntimeError('the number of croped features {} is not equal to the length of tracklet {}'.
                               format(num_crops, tracklet_length))
        label = torch.FloatTensor([label])

        return crop_f, label



















