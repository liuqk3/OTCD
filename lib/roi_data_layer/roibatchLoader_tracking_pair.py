
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from lib.model.utils.config import cfg
from lib.roi_data_layer.minibatch_tracking_pair import get_minibatch, get_minibatch
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb


def crop_the_data(data, gt_boxes, ratio):
    data_height, data_width = data.size(0), data.size(1)
    if ratio < 1:
        # this means that data_width << data_height, we need to crop the
        # data_height
        min_y = int(torch.min(gt_boxes[:, 1]))
        max_y = int(torch.max(gt_boxes[:, 3]))

        trim_size = int(np.floor(data_width / ratio))
        if trim_size > data_height:
            trim_size = data_height
        box_region = max_y - min_y + 1
        if min_y == 0:
            y_s = 0
        else:
            if (box_region - trim_size) < 0:
                y_s_min = max(max_y - trim_size, 0)
                y_s_max = min(min_y, data_height - trim_size)
                if y_s_min == y_s_max:
                    y_s = y_s_min
                else:
                    y_s = np.random.choice(range(y_s_min, y_s_max))
            else:
                y_s_add = int((box_region - trim_size) / 2)
                if y_s_add == 0:
                    y_s = min_y
                else:
                    y_s = np.random.choice(range(min_y, min_y + y_s_add))
        # crop the image
        data = data[y_s:(y_s + trim_size), :, :]

        # shift y coordiante of gt_boxes
        gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
        gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

        # update gt bounding box according the trip
        gt_boxes[:, 1].clamp_(0, trim_size - 1)
        gt_boxes[:, 3].clamp_(0, trim_size - 1)

    else:
        # this means that data_width >> data_height, we need to crop the
        # data_width
        min_x = int(torch.min(gt_boxes[:, 0]))  # (x1, y1, x2, y2)
        max_x = int(torch.max(gt_boxes[:, 2]))
        trim_size = int(np.ceil(data_height * ratio))
        if trim_size > data_width:
            trim_size = data_width
        box_region = max_x - min_x + 1
        if min_x == 0:
            x_s = 0
        else:
            if (box_region - trim_size) < 0:  # we try to crop all the boxes
                x_s_min = max(max_x - trim_size, 0)
                x_s_max = min(min_x, data_width - trim_size)
                if x_s_min == x_s_max:
                    x_s = x_s_min
                else:
                    x_s = np.random.choice(range(x_s_min, x_s_max))

            else:  # all boxes can not be cropped
                x_s_add = int((box_region - trim_size) / 2)
                if x_s_add == 0:
                    x_s = min_x
                else:
                    x_s = np.random.choice(range(min_x, min_x + x_s_add))
        # crop the image
        data = data[:, x_s:(x_s + trim_size), :]

        # shift x coordiante of gt_boxes
        gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
        gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
        # update gt bounding box according the trip
        gt_boxes[:, 0].clamp_(0, trim_size - 1)
        gt_boxes[:, 2].clamp_(0, trim_size - 1)

    return data, gt_boxes


def padding_the_data(data, im_info, gt_boxes, ratio):
    """
    This function padding the image based on the ratio
    :param data: 3D tensor, h x w x c
    :param im_info: 2D tensor [1, 3], h, w, resized_scale
    :param gt_boxes: 2D tensor, [num_box, 5], each row is (x1, y1, x2, y2 ,class_id)
    :param ratio: im_w / im_h
    :return:
    """

    data_height, data_width, data_channel = data.size(0), data.size(1), data.size(2)

    if ratio < 1:
        # this means that data_width < data_height
        trim_size = int(np.floor(data_width / ratio))

        padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), data_width, data_channel).zero_()

        padding_data[:data_height, :, :] = data
        # update im_info
        im_info[0, 0] = padding_data.size(0)
        # print("height %d %d \n" %(index, anchor_idx))
    elif ratio > 1:
        # this means that data_width > data_height
        # if the image need to crop.
        padding_data = torch.FloatTensor(data_height, \
                                         int(np.ceil(data_height * ratio)), data_channel).zero_()
        padding_data[:, :data_width, :] = data
        im_info[0, 1] = padding_data.size(1)
    else:
        trim_size = min(data_height, data_width)
        padding_data = data[:trim_size, :trim_size, :]
        # gt_boxes.clamp_(0, trim_size)
        gt_boxes[:, :4].clamp_(0, trim_size)
        im_info[0, 0] = trim_size
        im_info[0, 1] = trim_size
    return padding_data, im_info, gt_boxes



class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size,
               num_classes, group_size=12, training=True,
               normalize=None, im_type=['mv', 'residual']):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.group_size = group_size
    self.data_size = len(self.ratio_list)
    self.im_type = im_type # the type of data to load

    # given the ratio_list, we want to make the ratio same for each batch.
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
    self.target_size_batch = torch.Tensor(self.data_size).zero_()

    num_batch = int(np.ceil(len(ratio_index) / batch_size))
    for i in range(num_batch):
        left_idx = i*batch_size
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1.0

        self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio

  def resize_batch(self):
      num_batch = int(np.ceil(len(self.ratio_index) / self.batch_size))
      for i in range(num_batch):
          left_idx = i * self.batch_size
          right_idx = min((i + 1) * self.batch_size - 1, self.data_size - 1)

          if self.training:
              scale_inds = np.random.randint(0, high=len(cfg.TRAIN.SCALES), size=1)
              target_size = cfg.TRAIN.SCALES[scale_inds[0]]
          else:
              target_size = cfg.TEST.SCALES[0]
          self.target_size_batch[left_idx:(right_idx + 1)] = target_size

  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
    else:
        #index_ratio = int(self.ratio_index[index])
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one

    minibatch_db = self._roidb[index_ratio]
    target_size = self.target_size_batch[index]

    # get the frame_type of frames
    frame_type = []  # to store the frame type of this pair of image, 0 denotes i frame, 1 denotes p frame
    for j in range(len(minibatch_db)):
        if minibatch_db[j]['frame_type'] == 'i_frame':
            frame_type.append(0)
        elif minibatch_db[j]['frame_type'] == 'p_frame':
            frame_type.append(1)
        else:
            raise ValueError('unrecognized frame type {}'.format(minibatch_db[j]['frame_type']))
    frame_type = torch.Tensor(frame_type)

    blobs = get_minibatch(minibatch_db, self._num_classes, target_size, self.group_size, im_type=self.im_type) # dictionary

    im_info = torch.from_numpy(blobs['im_info']) # size [1, 3]

    if self.training:
        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################

        # NOTE1: need to cope with the case where a group cover both conditions. (done)
        # NOTE2: need to consider the situation for the tail samples. (no worry)
        # NOTE3: need to implement a parallel data loader. (no worry)
        # get the index range

        # if the image need to crop, crop to the target size.
        ratio = self.ratio_list_batch[index]
        need_crope = self._roidb[index_ratio][0]['need_crop']

        i_frame= blobs['data'][0]
        data_height, data_width = i_frame.shape[0], i_frame.shape[1]

        blobs['num_boxes'] = []
        for j in range(len(blobs['data'])):

            one_data = torch.from_numpy(blobs['data'][j]).contiguous() # [h, w, c]

            # remove the image, just keep the mv and residual
            one_data = one_data[:, :, 3:8].contiguous()

            gt_boxes = blobs['boxes'][j]
            #np.random.shuffle(gt_boxes)
            gt_boxes = torch.from_numpy(gt_boxes)

            # crop the data
            if self._roidb[index_ratio][j]['need_crop'] != need_crope:
                raise RuntimeError('The images in one group do not have same need_crop value.')

            if need_crope:
                one_data, gt_boxes = crop_the_data(one_data, gt_boxes, ratio)

            # padding the data
            # padding_data with size [height, width, c]
            padding_data, im_info, gt_boxes = padding_the_data(one_data, im_info, gt_boxes, ratio)

            # check the bounding box:
            not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
            keep = torch.nonzero(not_keep == 0).view(-1)

            gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
            if keep.numel() != 0:
                gt_boxes = gt_boxes[keep]
                num_boxes = min(gt_boxes.size(0), self.max_num_box)
                gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]
            else:
                num_boxes = 0

            padding_data = padding_data.permute(2, 0, 1).contiguous()

            # return the data back to blob
            blobs['data'][j] = padding_data
            blobs['num_boxes'].append(num_boxes)
            blobs['boxes'][j] = gt_boxes_padding
            blobs['im_info'] = im_info.view(3)

        im_info = blobs['im_info'] # size [3]
        im_info = torch.cat((im_info, frame_type), dim=0) # size [5]

        frame_1 = blobs['data'][0] # [c, h, w]
        frame_1_box = blobs['boxes'][0]
        num_box_1 = blobs['num_boxes'][0]

        frame_2 = blobs['data'][1]
        frame_2_box = blobs['boxes'][1]
        num_box_2 = blobs['num_boxes'][1]

        # filter the box, i.e. just keep the shared box
        # in order to do tracking regression, we need to keep the targets that appear in both frames
        target_id_1 = frame_1_box[:, 5]
        target_id_2 = frame_2_box[:, 5]
        target_id_share = []
        num_target_1 = target_id_1.size()[0]
        for idx in range(num_target_1):
            if target_id_1[idx] in target_id_2 and target_id_1[idx] != 0:
                target_id_share.append(target_id_1[idx])

        num_target_share = len(target_id_share)
        if num_target_share == 0:
            a = 1
            print('No shared targets in both frames!')

        frame_1_box_filter = frame_1_box.new(frame_1_box.size()).zero_()
        frame_2_box_filter = frame_2_box.new(frame_1_box.size()).zero_()
        for idx in range(num_target_share):
            idx1 = target_id_1 == target_id_share[idx]
            idx1 = torch.nonzero(idx1).squeeze()
            idx2 = target_id_2 == target_id_share[idx]
            idx2 = torch.nonzero(idx2).squeeze()

            frame_1_box_filter[idx, :] = frame_1_box[idx1].squeeze()
            frame_2_box_filter[idx, :] = frame_2_box[idx2].squeeze()

        num_box_1 = num_target_share
        num_box_2 = num_target_share

        # # show the data to see whether it has processed correctly
        # from lib.utils import offsets_to_coordinates, coordinates_to_flow_field
        # import matplotlib.pyplot as plt
        # import torch.nn.functional as F
        #
        # i_frame = i_frame.permute(1, 2, 0)
        # p_frame = p_frame.permute(1, 2, 0)
        #
        # i_im = i_frame[:,:,0:3].numpy() + 125
        # plt.imshow(np.asarray(i_im, dtype=np.uint8))
        # plt.show()
        #
        # coord = offsets_to_coordinates(p_frame[:,:, 3:5].numpy())
        # flow_filed = coordinates_to_flow_field(coord)
        #
        # p_im_warp = F.grid_sample(i_im, flow_filed)
        # r = p_frame[:,:,5:8].numpy()
        # p_im_warp = p_im_warp + r
        # plt.imshow(np.asarray(p_im_warp, dtype=np.uint8))
        # plt.show()

        return im_info, frame_1, frame_1_box_filter, num_box_1, frame_2, frame_2_box_filter, num_box_2
    else:
        # # ------------ show some results ------------------------
        # from lib.model.utils.misc import show_compressed_frame
        # show_compressed_frame(np.array(blobs['data'][1][:, :, 0:3] + cfg.PIXEL_MEANS, dtype=np.uint8), 0)
        # show_compressed_frame(blobs['data'][1][:, :, 3:5], 1)
        # show_compressed_frame(np.array(blobs['data'][1][:, :, 5:8], dtype=np.uint8), 2)

        raise NotImplementedError

        im_info = im_info.view(3)
        im_info = torch.cat((im_info, frame_type), dim=0)  # size [5]

        frame_data = blobs['data']
        i_frame = frame_data[0][:, :, 0:3] # for i frame, we only keep the first 3 channels, the other channels are all 0
        data_height, data_width, data_channel = i_frame.shape[0], i_frame.shape[1], i_frame.shape[2]
        i_frame = torch.from_numpy(i_frame).permute(2, 0, 1).contiguous().view(data_channel, data_height, data_width)
        i_frame_box = torch.FloatTensor([1,1,1,1,1])
        i_num_box = 0

        p_frame = frame_data[1]

        data_height, data_width, data_channel = p_frame.shape[0], p_frame.shape[1], p_frame.shape[2]
        p_frame = torch.from_numpy(p_frame).permute(2, 0, 1).contiguous().view(data_channel, data_height, data_width)
        p_frame_box = torch.FloatTensor([1,1,1,1,1])
        p_num_box = 0

        return im_info, i_frame, i_frame_box, i_num_box, p_frame, p_frame_box, p_num_box

  def __len__(self):
    return len(self._roidb)
