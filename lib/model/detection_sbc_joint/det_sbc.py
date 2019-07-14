import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.psroi_pooling.modules.psroi_pool_test import PSRoIPool
from lib.model.rpn.proposal_target_layer_cascade_gt_index import _ProposalTargetLayer
from lib.model.rpn.rpn import _RPN
from lib.model.utils.config import cfg
from lib.model.utils.net_utils import _smooth_l1_loss
from torch.autograd import Variable
import pdb
import math
import numpy as np
from lib.model.sbc_net.spatial_binary_classifier import SBC
from lib.model.detection_net.rfcn import RFCN
from lib.datasets.tools.misc import jitter_tracklets
from lib.model.roi_align.roi_align.roi_align import RoIAlign

class DET_SBC(nn.Module):

    # TODO now noly 101 detection net is supported!

    def __init__(self, det_num_layers=101, sbc_crop_size=(1024, 7, 7), pretrained=True):
        super(DET_SBC, self).__init__()

        self.sbc_crop_size = sbc_crop_size
        self.roi_align = RoIAlign(crop_width=sbc_crop_size[2], crop_height=sbc_crop_size[1], transform_fpcoor=True)

        self.sbc_model = SBC(input_h=sbc_crop_size[1], input_w=sbc_crop_size[2], input_c=sbc_crop_size[0])
        self.detection_model = RFCN(classes=('__background__', 'person'), base_net='resnet',
                                    num_layers_i=det_num_layers, class_agnostic=False, pretrained=pretrained)
        self.detection_model.create_architecture()

        self.data_type = None # 'single' or 'pair'
        # if 'single', this means the inputed images are independent, so we generate the appearance pairs
        # for appearance cnn in the batch based on each image independently.
        # if 'pair', this means the inputed images are nearby, so they may share the same targets. Hence,
        # we generate appearance pairs for appearance cnn based on the nearby images.


    def load_weights(self, sbc_weights, det_weights):
        """
        This function load the weight for the module.
        :param sbc_weights:
        :param det_weights:
        :return:
        """
        if sbc_weights is not None:
            self.sbc_model.load_state_dict(sbc_weights, strict=True)
        if det_weights is not None:
            self.detection_model.load_state_dict(det_weights, strict=True)

    def train(self, mode=True):
        self.training = mode
        if mode:
            self.sbc_model.train()
            self.detection_model.set_train_and_test_configure(phase='train', config='joint_with_sbc')
        else:
            self.sbc_model.eval()
            self.detection_model.set_train_and_test_configure(phase='train')

    def set_train_configure(self, mode=True, data_type=None):
        self.train(mode)
        self.data_type = data_type


    def eval(self):
        self.train(False)

    def _get_data_for_sbc(self, gt_boxes, num_boxes, base_feat, f_scale):
        """
        This function prepare the input data for sbc.
        :param gt_boxes: 2D tensor, bs x num_box x 5, each row is (x1, y1, x2, y2, class_id),
                        maybe bs x num_box x 6, (x1, y1, x2, y2, class_id., target_id)
        :param num_boxes: 2D tensor, bs x 1, the number of gt boxes of input frames
        :param base_feat: 4D tensor, [bs, c, h, w], te output of detection model
        :param f_scale: 1D , the scale used to map the gt boxes to feature map, [w_scale, h_scale, w_scale, h_scale]
        :return:
        """

        if base_feat.size(1) != self.sbc_crop_size[0]:
            raise RuntimeError('The channels of base feature output by detection net is {}, but expected is {}'.
                               format(base_feat.size(1), self.sbc_crop_size[0]))

        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data.int()

        max_num_pair = 64
        iou_up = 0.7

        if self.data_type == 'single':

            batch_size = gt_boxes.size(0)

            # store the first box
            pair_box_1 = gt_boxes.new(0, 5).zero_() # [x1, y1, x2, y2, batch_idx]
            # store the second box
            pair_box_2 = gt_boxes.new(0, 5).zero_()
            labels = gt_boxes.new(0).fill_(-1)

            for b in range(batch_size):

                if num_boxes[b] > 0:
                    box_tmp = gt_boxes[b, 0:num_boxes[b], 0:4]
                    batch_index = box_tmp.new(box_tmp.size(0), 1).fill_(b)
                    box_tmp = torch.cat((box_tmp, batch_index), dim=1)

                    gt_boxes_tmp = box_tmp # [N, 5], (x1, y1, x2, y2, batch_index)

                    # jitter all boxes
                    gt_boxes_np = np.asarray(gt_boxes_tmp[:, 0:4])  # [N, 4], [x1, y1, x2, y2]
                    positive_np_1 = jitter_tracklets(gt_boxes_np.copy(), iou_thr=iou_up, up_or_low='up', format='tlbr')
                    positive_np_2 = jitter_tracklets(gt_boxes_np.copy(), iou_thr=iou_up, up_or_low='up', format='tlbr')

                    gt_boxes_tmp_positive_1 = gt_boxes_tmp.clone()
                    gt_boxes_tmp_positive_1[:, 0:4] = gt_boxes_tmp.new(gt_boxes_tmp.size(0), 4).copy_(
                        torch.FloatTensor(positive_np_1))

                    gt_boxes_tmp_positive_2 = gt_boxes_tmp.clone()
                    gt_boxes_tmp_positive_2[:, 0:4] = gt_boxes_tmp.new(gt_boxes_tmp.size(0), 4).copy_(
                        torch.FloatTensor(positive_np_2))

                    negative_np = jitter_tracklets(gt_boxes_np.copy(), iou_thr=1 - iou_up, up_or_low='low', format='tlbr')
                    gt_boxes_tmp_negative = gt_boxes_tmp.clone()
                    gt_boxes_tmp_negative[:, 0:4] = gt_boxes_tmp.new(gt_boxes_tmp.size(0), 4).copy_(
                        torch.FloatTensor(negative_np))

                    # # compute the euclidean boxes. we use the jittered box
                    # gt_ct = gt_boxes_tmp_positive_1[:, 0:2] - 0.5 * gt_boxes_tmp_positive_1[:, 2:4] + 0.5  # [N, 2]
                    # gt_ct_1 = gt_ct.unsqueeze(dim=0)  # [1, N, 2]
                    # gt_ct_2 = gt_ct.unsqueeze(dim=1)  # [N, 1, 2]
                    # dist = gt_ct_1 - gt_ct_2  # [N, N, 2]
                    # dist = torch.norm(dist, p=2, dim=2)  # [N, N]
                    # dist_diag = torch.diag(dist)  # []
                    # dist_diag[dist_diag == 0] = 1e5
                    # dist = dist + torch.diag(dist_diag)  # set the diag large
                    #
                    # _, nearest_index = torch.min(dist, dim=1)  # min_dist, index all 1D tensor

                    # generate pairs
                    if num_boxes[b] >= 2:
                        # store the first box
                        pair_box_1_tmp = gt_boxes_tmp.new(num_boxes[b] * 4, 5).zero_()  # [x1, y1, x2, y2, batch_idx]
                        # store the second box
                        pair_box_2_tmp = gt_boxes_tmp.new(num_boxes[b] * 4, 5).zero_()
                        labels_tmp = gt_boxes_tmp.new(num_boxes[b] * 4).fill_(-1)

                        gt_box_count = 0
                        for i in range(num_boxes[b]):
                            count = gt_box_count * 4

                            # two positive pairs
                            pair_box_1_tmp[count, :] = gt_boxes_tmp[gt_box_count, :]
                            pair_box_2_tmp[count, :] = gt_boxes_tmp_positive_1[gt_box_count, :]
                            labels_tmp[count] = 1

                            pair_box_1_tmp[count + 1, :] = gt_boxes_tmp[gt_box_count, :]
                            pair_box_2_tmp[count + 1, :] = gt_boxes_tmp_positive_2[gt_box_count, :]
                            labels_tmp[count + 1] = 1

                            # two negative pairs
                            pair_box_1_tmp[count + 2, :] = gt_boxes_tmp[gt_box_count, :]
                            pair_box_2_tmp[count + 2, :] = gt_boxes_tmp_negative[gt_box_count, :]
                            labels_tmp[count + 2] = 0

                            pair_box_1_tmp[count + 3, :] = gt_boxes_tmp_positive_1[gt_box_count, :]
                            # pair_box_2_tmp[count + 3, :] = gt_boxes_tmp_positive_1[nearest_index[gt_box_count], :]
                            i_random = np.random.choice(np.array(list(range(i)) + list(range(i + 1, num_boxes[b]))))
                            pair_box_2_tmp[count + 3, :] = gt_boxes_tmp_positive_1[i_random, :]
                            labels_tmp[count + 3] = 0

                            gt_box_count += 1

                        # if b == 0:
                        #     pair_box_1 = pair_box_1_tmp.clone()
                        #     pair_box_2 = pair_box_2_tmp.clone()
                        #     labels = labels_tmp.clone()
                        # else:
                        #     pair_box_1 = torch.cat((pair_box_1, pair_box_1_tmp), dim=0)
                        #     pair_box_2 = torch.cat((pair_box_2, pair_box_2_tmp), dim=0)
                        #     labels = torch.cat((labels, labels_tmp), dim=0)

                    elif num_boxes[b] == 1: # in only one box
                        # store the first box
                        pair_box_1_tmp = gt_boxes_tmp.new(num_boxes[b] * 2, 5).zero_()  # [x1, y1, x2, y2, batch_idx]
                        # store the second box
                        pair_box_2_tmp = gt_boxes_tmp.new(num_boxes[b] * 2, 5).zero_()
                        labels_tmp = gt_boxes_tmp.new(num_boxes[b] * 2).fill_(-1)

                        gt_box_count = 0
                        for i in range(num_boxes[b]):
                            count = gt_box_count * 2

                            # positive pairs
                            pair_box_1_tmp[count, :] = gt_boxes_tmp[gt_box_count, :]
                            pair_box_2_tmp[count, :] = gt_boxes_tmp_positive_1[gt_box_count, :]
                            labels_tmp[count] = 1

                            # negative pairs
                            pair_box_1_tmp[count + 1, :] = gt_boxes_tmp[gt_box_count, :]
                            pair_box_2_tmp[count + 1, :] = gt_boxes_tmp_negative[gt_box_count, :]
                            labels_tmp[count + 1] = 0

                            gt_box_count += 1

                    if b == 0:
                        pair_box_1 = pair_box_1_tmp.clone()
                        pair_box_2 = pair_box_2_tmp.clone()
                        labels = labels_tmp.clone()
                    else:
                        pair_box_1 = torch.cat((pair_box_1, pair_box_1_tmp), dim=0)
                        pair_box_2 = torch.cat((pair_box_2, pair_box_2_tmp), dim=0)
                        labels = torch.cat((labels, labels_tmp), dim=0)

            num_pair = min(labels.size(0), max_num_pair)
            # there is no need to shuffle the pairs, since the boxes are shuffled in the dataloader
            # and the positive and negative pairs are uniformly distributed
            pair_box_1 = pair_box_1[0:num_pair]
            pair_box_2 = pair_box_2[0:num_pair]
            labels = labels[0:num_pair]

            pair_box_1 = Variable(pair_box_1)
            pair_box_2 = Variable(pair_box_2)
            labels = Variable(labels)
            f_scale = Variable(f_scale.unsqueeze(dim=0))

            crop_f_1 = self.roi_align(base_feat, pair_box_1[:, 0:4] * f_scale, pair_box_1[:, 4].int())
            crop_f_2 = self.roi_align(base_feat, pair_box_2[:, 0:4] * f_scale, pair_box_2[:, 4].int())

            return crop_f_1, crop_f_2, labels

        elif self.data_type == 'pair':
            # if the batch is a pair of nearby two frames:
            # For each gt box, we generate 4 pairs:
            # 1) one positive: if the target exists in two frames, treat the two boxes of one target as
            #   positive, else jittered from it self (iou > 0.7)
            # 2) one negative: the nearest other box

            # get the feature index for each boxes
            batch_size = gt_boxes.size(0)
            gt_boxes_tmp = gt_boxes.clone()
            for b in range(batch_size):
                gt_boxes_tmp[b, :, 4] = b

            # firstly divide the gt_boxes, num_boxes
            bs = int(batch_size / 2)

            gt_boxes_1 = gt_boxes_tmp[0:bs, :, :].contiguous()
            gt_boxes_2 = gt_boxes_tmp[bs:, :, :].contiguous()
            if gt_boxes_1.size(0) != gt_boxes_2.size(0):
                raise RuntimeError('The number of first images is not the same with the second images!')

            num_boxes_1 = num_boxes[0:bs].contiguous()
            num_boxes_2 = num_boxes[bs:].contiguous()

            pair_box_1 = gt_boxes.new(0, 5).zero_() # each row is (x1, y1, x2, y2, base_feature_index)
            pair_box_2 = gt_boxes.new(0, 5).zero_()
            labels = gt_boxes.new(0).fill_(-1)
            for b in range(bs):

                gt_boxes_1_tmp = gt_boxes_1[b] # [num_boxes, 6], [x1, y1, x2, y2, base_feat_idx, target_id]
                gt_boxes_2_tmp = gt_boxes_2[b]

                # jitter the boxes
                if num_boxes_1[b] > 0:
                    box_np = np.asarray(gt_boxes_1_tmp[0:num_boxes_1[b], 0:4])

                    box_jitter = jitter_tracklets(box_np.copy(), iou_thr=iou_up, up_or_low='up', format='tlbr')
                    gt_boxes_1_tmp_positive = gt_boxes_1_tmp.clone()
                    gt_boxes_1_tmp_positive[0:num_boxes_1[b], 0:4] = gt_boxes_1_tmp.new(num_boxes_1[b], 4).copy_(
                            torch.FloatTensor(box_jitter))

                    # box_jitter = jitter_tracklets(box_np.copy(), iou_thr=1 - iou_up, up_or_low='low', format='tlbr')
                    # gt_boxes_1_tmp_negative = gt_boxes_1_tmp.clone()
                    # gt_boxes_1_tmp_negative[0:num_boxes_1[b], 0:4] = gt_boxes_1_tmp.new(num_boxes_1[b], 4).copy_(
                    #     torch.FloatTensor(box_jitter))
                else:
                    gt_boxes_1_tmp_positive = gt_boxes_1_tmp.clone()
                    # gt_boxes_1_tmp_negative = gt_boxes_1_tmp.clone()

                if num_boxes_2[b] > 0:
                    box_np = np.asarray(gt_boxes_2_tmp[0:num_boxes_2[b], 0:4])

                    box_jitter = jitter_tracklets(box_np.copy(), iou_thr=iou_up, up_or_low='up', format='tlbr')
                    gt_boxes_2_tmp_positive = gt_boxes_2_tmp.clone()
                    gt_boxes_2_tmp_positive[0:num_boxes_2[b], 0:4] = gt_boxes_2_tmp.new(num_boxes_2[b], 4).copy_(
                        torch.FloatTensor(box_jitter))

                    # box_jitter = jitter_tracklets(box_np.copy(), iou_thr=1 - iou_up, up_or_low='low', format='tlbr')
                    # gt_boxes_2_tmp_negative = gt_boxes_2_tmp.clone()
                    # gt_boxes_2_tmp_negative[0:num_boxes_2[b], 0:4] = gt_boxes_2_tmp.new(num_boxes_2[b], 4).copy_(
                    #     torch.FloatTensor(box_jitter))
                else:
                    gt_boxes_2_tmp_positive = gt_boxes_2_tmp.clone()
                    # gt_boxes_2_tmp_negative = gt_boxes_2_tmp.clone()

                target_id_1_tmp = gt_boxes_1_tmp[:, 5] # [num_box]
                target_id_2_tmp = gt_boxes_2_tmp[:, 5]

                # -------------------------------- generate positive pairs -------------------------------------
                # find the shared targets
                shared_targets = []
                unshared_targets_1 = []
                unshared_targets_2 = []
                for box_idx in range(num_boxes_1[b]):
                    target_id = target_id_1_tmp[box_idx]
                    if target_id > 0:
                        if target_id in target_id_2_tmp:
                            shared_targets.append(target_id_1_tmp[box_idx])
                        else:
                            unshared_targets_1.append((target_id))
                for box_idx in range(num_boxes_2[b]):
                    target_id = target_id_2_tmp[box_idx]
                    if target_id > 0:
                        if target_id not in shared_targets:
                            unshared_targets_2.append(target_id)

                # generate pairs for shared targets. All boxes are the jittered box
                pair_box_1_share_tmp = gt_boxes.new(len(shared_targets), 5).zero_() # [x1, y1, x2, y2, base_feat_idx]
                pair_box_2_share_tmp = gt_boxes.new(len(shared_targets), 5).zero_()
                labels_share_tmp = gt_boxes.new(len(shared_targets)).fill_(-1)
                for idx in range(len(shared_targets)):
                    target_id = shared_targets[idx]

                    index_1 = gt_boxes_1_tmp_positive[:, 5] == target_id
                    index_1 = torch.nonzero(index_1).squeeze()
                    pair_box_1_share_tmp[idx, :] = gt_boxes_1_tmp_positive[index_1].squeeze()[0:5]

                    index_2 = gt_boxes_2_tmp_positive[:, 5] == target_id
                    index_2 = torch.nonzero(index_2).squeeze()
                    pair_box_2_share_tmp[idx, :] = gt_boxes_2_tmp_positive[index_2, :].squeeze()[0:5]

                    labels_share_tmp[idx] = 1

                # generate pairs for unshared targets in gt_boxes_1
                # among two boxes in one pair, one is the gt box, the other is jitterd box
                pair_box_1_unshared_1_tmp = gt_boxes.new(len(unshared_targets_1), 5).zero_() # [x1, y1, x2, y2, base_feat_idx]
                pair_box_2_unshared_1_tmp = gt_boxes.new(len(unshared_targets_1), 5).zero_()
                labels_unshared_1_tmp = gt_boxes.new(len(unshared_targets_1)).fill_(-1)
                for idx in range(len(unshared_targets_1)):
                    target_id = unshared_targets_1[idx]

                    index_1 = gt_boxes_1_tmp_positive[:, 5] == target_id
                    index_1 = torch.nonzero(index_1).squeeze()
                    pair_box_1_unshared_1_tmp[idx, :] = gt_boxes_1_tmp[index_1].squeeze()[0:5]
                    pair_box_2_unshared_1_tmp[idx, :] = gt_boxes_1_tmp_positive[index_1].squeeze()[0:5]
                    labels_unshared_1_tmp[idx] = 1

                # generate paires for unshared targets in gt_boxes_2
                # among two boxes in one pair, one is the gt box, the other is jitterd box
                pair_box_1_unshared_2_tmp = gt_boxes.new(len(unshared_targets_2), 5).zero_() # [x1, y1, x2, y2, base_feat_idx]
                pair_box_2_unshared_2_tmp = gt_boxes.new(len(unshared_targets_2), 5).zero_()
                labels_unshared_2_tmp = gt_boxes.new(len(unshared_targets_2)).fill_(-1)
                for idx in range(len(unshared_targets_2)):
                    target_id = unshared_targets_2[idx]

                    index_2 = gt_boxes_2_tmp_positive[:, 5] == target_id
                    index_2 = torch.nonzero(index_2).squeeze()
                    pair_box_1_unshared_2_tmp[idx, :] = gt_boxes_2_tmp[index_2].squeeze()[0:5]
                    pair_box_2_unshared_2_tmp[idx, :] = gt_boxes_2_tmp_positive[index_2].squeeze()[0:5]
                    labels_unshared_2_tmp[idx] = 1

                # -------------------------------- generate negative pairs -------------------------------------
                # generate negative pairs for gt_box_1_tmp
                # compute the euclidean distance between boxes. we use the jittered box
                gt_ct = gt_boxes_1_tmp_positive[:, 0:2] - 0.5 * gt_boxes_1_tmp_positive[:, 2:4] + 0.5  # [N, 2]
                gt_ct_1 = gt_ct.unsqueeze(dim=0)  # [1, N, 2]
                gt_ct_2 = gt_ct.unsqueeze(dim=1)  # [N, 1, 2]
                dist = gt_ct_1 - gt_ct_2  # [N, N, 2]
                dist = torch.norm(dist, p=2, dim=2)  # [N, N]
                dist_diag = torch.diag(dist)  # []
                dist_diag[dist_diag == 0] = 1e5
                dist = dist + torch.diag(dist_diag)  # set the diag large

                _, nearest_index = torch.min(dist, dim=1)  # min_dist, index all 1D tensor

                pair_box_1_negative_1_tmp = gt_boxes.new(num_boxes_1[b], 5).zero_() # [x1, y1, x2, y2, base_feat_idx]
                pair_box_2_negative_1_tmp = gt_boxes.new(num_boxes_1[b], 5).zero_()
                labels_negative_1_tmp = gt_boxes.new(num_boxes_1[b]).fill_(-1)
                for idx in range(num_boxes_1[b]):
                    pair_box_1_negative_1_tmp[idx, :] = gt_boxes_1_tmp_positive[idx].squeeze()[0:5]
                    pair_box_2_negative_1_tmp[idx, :] = gt_boxes_1_tmp_positive[nearest_index[idx]].squeeze()[0:5]
                    labels_negative_1_tmp[idx] = 0

                # generate negative pairs for gt_box_2_tmp
                # compute the euclidean distance between boxes. we use the jittered box
                gt_ct = gt_boxes_2_tmp_positive[:, 0:2] - 0.5 * gt_boxes_2_tmp_positive[:, 2:4] + 0.5  # [N, 2]
                gt_ct_1 = gt_ct.unsqueeze(dim=0)  # [1, N, 2]
                gt_ct_2 = gt_ct.unsqueeze(dim=1)  # [N, 1, 2]
                dist = gt_ct_1 - gt_ct_2  # [N, N, 2]
                dist = torch.norm(dist, p=2, dim=2)  # [N, N]
                dist_diag = torch.diag(dist)  # []
                dist_diag[dist_diag == 0] = 1e5
                dist = dist + torch.diag(dist_diag)  # set the diag large

                _, nearest_index = torch.min(dist, dim=1)  # min_dist, index all 1D tensor

                pair_box_1_negative_2_tmp = gt_boxes.new(num_boxes_2[b], 5).zero_()  # [x1, y1, x2, y2, base_feat_idx]
                pair_box_2_negative_2_tmp = gt_boxes.new(num_boxes_2[b], 5).zero_()
                labels_negative_2_tmp = gt_boxes.new(num_boxes_2[b]).fill_(-1)
                for idx in range(num_boxes_2[b]):
                    pair_box_1_negative_2_tmp[idx, :] = gt_boxes_2_tmp_positive[idx].squeeze()[0:5]
                    pair_box_2_negative_2_tmp[idx, :] = gt_boxes_2_tmp_positive[nearest_index[idx]].squeeze()[0:5]
                    labels_negative_2_tmp[idx] = 0

                # concate all pairs
                pair_box_1_tmp = torch.cat((pair_box_1_share_tmp,
                                            pair_box_1_unshared_1_tmp,
                                            pair_box_1_unshared_2_tmp,
                                            pair_box_1_negative_1_tmp,
                                            pair_box_1_negative_2_tmp), dim=0)
                pair_box_2_tmp = torch.cat((pair_box_2_share_tmp,
                                            pair_box_2_unshared_1_tmp,
                                            pair_box_2_unshared_2_tmp,
                                            pair_box_2_negative_1_tmp,
                                            pair_box_1_negative_2_tmp), dim=0)
                labels_tmp = torch.cat((labels_share_tmp,
                                       labels_unshared_1_tmp,
                                       labels_unshared_2_tmp,
                                       labels_negative_1_tmp,
                                       labels_negative_2_tmp), dim=0)

                pair_box_1 = torch.cat((pair_box_1, pair_box_1_tmp), dim=0)
                pair_box_2 = torch.cat((pair_box_2, pair_box_2_tmp), dim=0)
                labels = torch.cat((labels, labels_tmp), dim=0)

            # shuffle the pairs
            pair = torch.cat((pair_box_1, pair_box_2, labels.unsqueeze(dim=1)), dim=1) # size [num_pair, 5 + 5 + 1]

            pair_np = np.asarray(pair)
            np.random.shuffle(pair_np)
            shuffled_pair = torch.FloatTensor(pair_np)
            pair = pair.new(pair.size()).copy_(shuffled_pair)

            pair_box_1 = pair[:, 0:5].contiguous()
            pair_box_2 = pair[:, 5:10].contiguous()
            labels = pair[:, 10].contiguous()

            num_pair = min(labels.size(0), max_num_pair)
            # there is no need to shuffle the pairs, since the boxes are shuffled in the dataloader
            # and the positive and negative pairs are uniformly distributed
            pair_box_1 = pair_box_1[0:num_pair]
            pair_box_2 = pair_box_2[0:num_pair]
            labels = labels[0:num_pair]

            pair_box_1 = Variable(pair_box_1)
            pair_box_2 = Variable(pair_box_2)
            labels = Variable(labels)
            f_scale = Variable(f_scale.unsqueeze(dim=0))

            crop_f_1 = self.roi_align(base_feat, pair_box_1[:, 0:4] * f_scale, pair_box_1[:, 4].int())
            crop_f_2 = self.roi_align(base_feat, pair_box_2[:, 0:4] * f_scale, pair_box_2[:, 4].int())

            return crop_f_1, crop_f_2, labels

    def forward(self, *input):
        # TODO: implement forwar for testing
        """
        input = im_info, i_frame, i_frame_box, i_num_box

        :param im_info: 2D tensor, bs x 3, [h, w, scale]
        :param i_frame: 4D tensor, bs x 8 x h x w, the 0:3 channel is image, 3:5 channel is motion vector,
                    5:8 channel is the residual
        :param i_frame_box: 2D tensor, bs x num_box x 5, each row is (x1, y1, x2, y2, class_id)
        :param i_num_box: 2D tensor, bs x 1, the number of gt boxes of i frame

        :param input:
        :return:
        """

        if self.data_type == None:
            raise RuntimeError('Please call function set_train_configure() first before training and testing!')

        if self.training:

            if self.data_type == 'single':
                # input = (im_info, im_data, gt_boxes, num_boxes)

                # the output of detection model contains:
                # rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls,
                # RCNN_loss_bbox, rois_label, base_feat_out, f_scale
                det_output = self.detection_model(*input)
                rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls,  RCNN_loss_bbox, rois_label, base_feat, f_scale = \
                    det_output[3:]

                gt_boxes, num_boxes = input[2], input[3]

                # the number of gt boxes maybe zero, hence we do not need to train sbc
                if num_boxes.sum().data[0] > 0:
                    f1, f2, labels = self._get_data_for_sbc(gt_boxes, num_boxes, base_feat, f_scale)
                    sbc_cls_score, _ = self.sbc_model(f1, f2)
                    sbc_cls_loss = self.sbc_model.get_loss(sbc_cls_score, labels.long(), smooth=True)
                else:
                    sbc_cls_loss = rpn_loss_cls.clone()
                    sbc_cls_loss[0] = 0

                return rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, sbc_cls_loss
            elif self.data_type == 'pair':
                # input = (im_info, im_data_1, gt_box_1, num_box_1, im_data_2, gt_box_2, num_box_2)
                im_info = input[0]
                im_data_1 = input[1]
                gt_boxes_1 = input[2]
                num_boxes_1 = input[3]

                im_data_2 = input[4]
                gt_boxes_2 = input[5]
                num_boxes_2 = input[6]

                # process the data to train the detector
                im_info = torch.cat((im_info, im_info), dim=0)
                im_data = torch.cat((im_data_1, im_data_2), dim=0)
                gt_boxes = torch.cat((gt_boxes_1, gt_boxes_2), dim=0)
                num_boxes = torch.cat((num_boxes_1, num_boxes_2), dim=0)

                # the output of detection model contains:
                # rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls,
                # RCNN_loss_bbox, rois_label, base_feat_out, f_scale
                det_output = self.detection_model(im_info, im_data, gt_boxes, num_boxes)
                rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls,  RCNN_loss_bbox, rois_label, base_feat, f_scale = \
                    det_output[3:]

                if num_boxes.sum().data[0] > 0:
                    f1, f2, labels = self._get_data_for_sbc(gt_boxes, num_boxes, base_feat, f_scale)
                    sbc_cls_score, _ = self.sbc_model(f1, f2)
                    sbc_cls_loss = self.sbc_model.get_loss(sbc_cls_score, labels.long(), smooth=True)
                else:
                    sbc_cls_loss = rpn_loss_cls.clone()
                    sbc_cls_loss[0] = 0
                return rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, sbc_cls_loss
        else:
            raise NotImplementedError




if __name__ == '__main__':
    print('runing rfcn_head...')