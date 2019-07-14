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
import numpy as np
import scipy.sparse
import pickle
from lib.datasets.imdb_pair import imdb
from lib.datasets.tools.misc import load_citypersons_annotation
from lib.datasets.imdb_single import ROOT_DIR

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from lib.model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete


class citypersons(imdb):
    def __init__(self, phase, name='citypersons_pair', cityperson_database_path=None):
        print('loading ' + name + '...')

        self._phase = phase

        imdb.__init__(self, name)

        # the group size in the raw mpeg4 video
        self._group_size = 12

        self._data_path = cityperson_database_path if cityperson_database_path is not None else '/home/liuqk/Dataset/Cityscapes/citysacpesdataset/'

        # there are only 2 classes in CityPersons. Noted that there are 6 categories in CityPersons:
        # ['pedestrian', 'ignore', 'person group', 'rider', 'person (other)', 'sitting person']
        self._classes = ('__background__',  # always index 0
                         'person')

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'

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
        :return: a list, each element in it is a string, i.e. jena_000078_000019_leftImg8bit/000001.png
        """
        if self._phase in ['train', 'training']:
            phase = 'train'
        elif self._phase in ['validation', 'val']:
            phase = 'val'
        elif self._phase in ['test', 'testing', 'inference']:
            phase = 'test'
        else:
            raise ValueError('Unrecgonized phase: {}'.format(self._phase))
        split = 'leftImg8bit'

        split_path = os.path.join(self._data_path, split, phase)
        cities = os.listdir(split_path)

        im_index = []

        for city in cities:
            city_dir = os.path.join(split_path, city)
            images = os.listdir(city_dir)
            # filter the .png images
            for im in images: # such as: jena_000078_000019_leftImg8bit.png
                im_list = im.split('.')

                # make sure this file is an image
                if im_list[-1] == self._image_ext[1:]:
                    frame_id1 = 1 # frame_id starts from 1
                    one_index1 = os.path.join(im_list[0], str(frame_id1).zfill(6) + self._image_ext)

                    # check if there exits fg in this image
                    ann_path = self._ann_path_from_index(one_index1)
                    im_info_path = self._im_info_path_from_index(one_index1)
                    gt_box, gt_classes = load_citypersons_annotation(ann_path, im_info_path, frame_id1,
                                                                     ambiguous_class_id=cfg.TRAIN.AMBIGUOUS_CLASS_ID,
                                                                     vis_threshold=cfg.TRAIN.VISIBILITY_THRESHOLD)
                    idx_fg = gt_classes == 1
                    fg_count = idx_fg.sum()
                    if fg_count > 0:
                        frame_id2 = 2
                        one_index2 = os.path.join(im_list[0], str(frame_id2).zfill(6) + self._image_ext)
                        im_index.append([one_index1, one_index2])
        return im_index

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        the index has the format:  jena_000078_000019_leftImg8bit/000001.png
        """
        if self._phase in ['train', 'training']:
            phase = 'train'
        elif self._phase in ['validation', 'val']:
            phase = 'val'
        elif self._phase in ['test', 'testing', 'inference']:
            phase = 'test'
        else:
            raise ValueError('Unrecgonized phase: {}'.format(self._phase))
        split = 'leftImg8bit'
        index_list = index.split('_')
        city = index_list[0]
        im_path = os.path.join(self._data_path, split, phase, city, index)
        if not os.path.exists(im_path):
            raise RuntimeError('The image path does not exist: {}'.format(im_path))

        return im_path

    def image_at_group_id(self, i):
        """
        return the group id of the frames
        :param i:
        :return:
        """
        return i

    def _ann_path_from_index(self, index):
        """
        Construct annotation path from the image's "index" identifier.
        the index has the format:  jena_000078_000019_leftImg8bit/000001.png
        """
        if self._phase in ['train', 'training']:
            phase = 'train'
        elif self._phase in ['validation', 'val']:
            phase = 'val'
        elif self._phase in ['test', 'testing', 'inference']:
            phase = 'test'
        else:
            raise ValueError('Unrecgonized phase: {}'.format(self._phase))

        split = 'gtBboxCityPersons'
        index_list = index.split('_')
        city = index_list[0]
        gt_name = index_list[0] + '_' + index_list[1] + '_' + index_list[2] + '_' + split + '.json'

        ann_path = os.path.join(self._data_path, split, phase, city, gt_name)

        if not os.path.exists(ann_path):
            raise RuntimeError('Annotation file path: {} does not exist!'.format(ann_path))

        return ann_path


    def _im_info_path_from_index(self, index):
        """
        Construct an im_info path from the image's "index" identifier.
        the index has the format:  jena_000078_000019_leftImg8bit/000001.png
        """
        if self._phase in ['train', 'training']:
            phase = 'train'
        elif self._phase in ['validation', 'val']:
            phase = 'val'
        elif self._phase in ['test', 'testing', 'inference']:
            phase = 'test'
        else:
            raise ValueError('Unrecgonized phase: {}'.format(self._phase))
        split = 'leftImg8bit'
        index_list = index.split('_')
        city = index_list[0]
        im_info_path = os.path.join(self._data_path, split, phase, city, index.split('/')[0], 'im_info.txt')
        if not os.path.exists(im_info_path):
            raise RuntimeError('The image path does not exist: {}'.format(im_info_path))

        return im_info_path

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        print('load gt boxes for all frames...')

        gt_roidb = []
        for i in range(len(self._image_index)):
            one_group_gt = [self._load_citypersons_annotation(index)
                        for index in self._image_index[i]]
            gt_roidb.append(one_group_gt)
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_citypersons_annotation(self, index):
        """
        the index has the format:  jena_000078_000019_leftImg8bit/000001.png
        """
        # ['pedestrian', 'ignore', 'person group', 'rider', 'person (other)', 'sitting person']
        print('load gt for {}'.format(index))

        ann_path = self._ann_path_from_index(index)
        im_info_path = self._im_info_path_from_index(index)
        frame_id = int(index.split('/')[-1].split('.')[0])

        gt_box, gt_classes = load_citypersons_annotation(ann_path, im_info_path, frame_id,
                                                         ambiguous_class_id=cfg.TRAIN.AMBIGUOUS_CLASS_ID,
                                                         vis_threshold=cfg.TRAIN.VISIBILITY_THRESHOLD)

        overlaps = np.zeros((gt_box.shape[0], self.num_classes), dtype=np.float32)
        overlaps[gt_classes==1, 1] = 1.0  # set person class boxes to the corresponding index is 1
        overlaps = scipy.sparse.csr_matrix(overlaps)
        frame_type = 'i_frame' if (frame_id-1) % self._group_size == 0 else 'p_frame'

        return {'boxes': gt_box,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'index': index,
                'frame_type': frame_type,
                'dataset_name': self.name
                }

    # some functions to evaluate with the metric of voc
    def _write_citypersons_results_file(self, all_boxes, output_dir='detect_results'):
        path = os.path.join(output_dir)
        if not os.path.exists(path):
            raise RuntimeError('No directory: ' + path)

        # write the detection results for each class
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} mot results file'.format(cls))
            filename = os.path.join(path, self.name + '_' + cls + '.txt')
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    # each line in the file is: index, confidence, x1, y1, x2, y2
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _citypersons_ap(self, rec, prec, use_07_metric=False):
        """ ap = _citypersons_ap(rec, prec, [use_07_metric])
        Compute mot AP given precision and recall.
        If voc_07_metric is true, uses the
        mvoc 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _citypersons_eval(self, detpath, classname, ovthresh=0.5, use_07_metric=False):
        """rec, prec, ap = voc_eval(detpath,
                                    classname,
                                    [ovthresh],
                                    [use_07_metric])

        Top level function that does the PASCAL VOC evaluation.

        detpath: Path to detections
        classname: Category name (duh)
        cachedir: Directory for caching the annotations
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file

        # first load gt
        # extract gt objects for this class
        class_recs = {}
        npos = 0

        gt_anno = self.gt_roidb()
        for idx in range(len(gt_anno)):
            anno = gt_anno[idx]
            imgindex = anno['index']

        # for imgindex in self._image_index:
        #     anno = self._load_citypersons_annotation(imgindex) # all the boxes are for person

            bbox = np.asarray([anno['boxes'][idx] for idx in range(anno['boxes'].shape[0]) if self._classes[anno['gt_classes'][idx]] == classname])

            difficult = np.zeros(bbox.shape[0]).astype(np.bool) # we treat all boxes as easy
            det = [False] * bbox.shape[0]
            npos = npos + sum(~difficult)
            class_recs[imgindex] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}
        # read dets
        with open(detpath, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]
        img_indices = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        nd = len(img_indices)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        if BB.shape[0] > 0:
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            img_indices = [img_indices[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            for d in range(nd):
                R = class_recs[img_indices[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self._citypersons_ap(rec, prec, use_07_metric)

        return rec, prec, ap


    def _do_python_eval(self,overthresh, output_dir='output', metric_year='2007'):

        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(metric_year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = os.path.join(output_dir, self.name + '_' + cls + '.txt')
            rec, prec, ap = self._citypersons_eval(filename, cls, ovthresh=overthresh, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.5f}'.format(ap))
        print('mean ap: {:.5f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir, overthresh=0.5):
        self._write_citypersons_results_file(all_boxes, output_dir)
        self._do_python_eval(output_dir=output_dir, overthresh=overthresh)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = os.path.join(output_dir, self.name + '_' + cls + '.txt')
                os.remove(filename)

if __name__ == '__main__':
    d = citypersons('motchallenge')
    res = d.roidb
    from IPython import embed

    embed()
