"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lib.datasets
import numpy as np
from lib.model.utils.config import cfg
from lib.datasets.factory import get_imdb
import PIL
import pdb


def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """

    roidb = imdb.roidb
    if not (imdb.name.startswith('coco')):
        sizes = []
        for i in range(len(imdb.image_index)):

            one_group_sizes = []
            last_s = (0, 0)
            for j in range(len(imdb.image_index[i])):
                s = PIL.Image.open(imdb.image_path_from_index(imdb.image_index[i][j])).size
                one_group_sizes.append(s)

                # make sure the size of images in one group are the same
                if j > 0 and last_s != s:
                    raise RuntimeError('the size of images in group ' + str(i) + ' are not the same...')
                last_s = s

            sizes.append(one_group_sizes)

    for i in range(len(imdb.image_index)):
        for j in range(len(imdb.image_index[i])):

            roidb[i][j]['group_id'] = imdb.image_at_group_id(i)
            roidb[i][j]['image'] = imdb.image_path_from_index(imdb.image_index[i][j])
            if not (imdb.name.startswith('coco')):
                roidb[i][j]['width'] = sizes[i][j][0]
                roidb[i][j]['height'] = sizes[i][j][1]
            # need gt_overlaps as a dense array for argmax
            gt_overlaps = roidb[i][j]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)
            roidb[i][j]['max_classes'] = max_classes
            roidb[i][j]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    # since the frames in one group has the same width and height,
    # we only need to handle the first frmae (I frame) in each
    # group

    def set_need_crop_tag(one_group_roidb, tag):
        for j in range(len(one_group_roidb)):
            one_group_roidb[j]['need_crop'] = tag

    ratio_large = 2.0  # largest ratio to preserve.
    ratio_small = 0.5  # smallest ratio to preserve.

    ratio_list = []
    for i in range(len(roidb)):
        width = roidb[i][0]['width']
        height = roidb[i][0]['height']
        ratio = width / float(height)

        if ratio > ratio_large:
            set_need_crop_tag(roidb[i], 1)
            ratio = ratio_large
        elif ratio < ratio_small:
            set_need_crop_tag(roidb[i], 1)
            ratio = ratio_small
        else:
            set_need_crop_tag(roidb[i], 0)

        ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index


def filter_roidb(roidb):
    # filter the image without bounding box.
    print('-----------filter the image without bounding box-------------')
    print('before filtering, there are %d groups...' % (len(roidb)))
    group_size = 0
    for i in range(len(roidb)):
        for j in range(len(roidb[i])):
            if len((roidb[i][j]['boxes'])) == 0:
                del roidb[i][j]
        # we treat the max length of group as the group size here
        if group_size < len(roidb[i]):
            group_size = len(roidb[i])

    # if the group length is less than group size, delete this group
    # for the easier batch loader
    roidb_f = []
    for group in roidb:
        if len(group) == group_size:
            roidb_f.append(group)

    print('after filtering, there are %d groups...' % (len(roidb_f)))
    return roidb_f


def combined_roidb(imdb_names, training=True):
    """
    Combine multiple roidbs
    """

    def get_training_roidb(imdb):
        """Returns a roidb (Region of Interest database) for use in training."""
        if cfg.TRAIN.USE_FLIPPED:
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_images()
            print('done')

        print('Preparing training data...')

        prepare_roidb(imdb)
        # ratio_index = rank_roidb_ratio(imdb)
        print('done')

        return imdb.roidb

    def get_roidb(imdb):
        # imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    imdb = get_imdb(imdb_names)
    roidb = get_roidb(imdb)

    # roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    # roidb = roidbs[0]
    #
    # if len(roidbs) > 1:
    #   for r in roidbs[1:]:
    #       roidb.extend(r)
    #   tmp = get_imdb(imdb_names.split('+')[1])
    #   imdb = lib.datasets.imdb(imdb_names, tmp.classes)
    # else:
    #   imdb = get_imdb(imdb_names)

    if training:
        roidb = filter_roidb(roidb)

    ratio_list, ratio_index = rank_roidb_ratio(roidb)

    return imdb, roidb, ratio_list, ratio_index
