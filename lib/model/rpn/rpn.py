from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from lib.model.utils.net_utils import _smooth_l1_loss, smooth_classification_loss

import numpy as np
import math
import pdb
import time


class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0], # batch
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def _iou_weighted_cross_entropy_loss(self, cls_score, roi_label, iou_weights):
        """

        :param cls_score: 2D tensor [N, C]
        :param roi_label: 1D tensor, [N], the label for each roi
        :param iou_weights: 1D tensor, [N], the weights for each roi
        :return:  the loss
        """
        num_rois, num_class = cls_score.size()[0], cls_score.size()[1]
        loss = - F.log_softmax(cls_score, 1)

        roi_label_expand = roi_label.unsqueeze(dim=1).repeat(1, num_class)

        class_id = torch.LongTensor(range(num_class)).unsqueeze(dim=0)
        class_id = class_id.repeat(num_rois, 1)

        if roi_label.data.is_cuda:
            class_id = class_id.cuda()
        class_id = Variable(class_id, requires_grad=False)

        mask = class_id == roi_label_expand
        loss = loss[mask]
        loss = loss * iou_weights

        return loss.mean()

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        # get the prob based on rpn_cls_score, but we need to reshape rpn_cls_score before softmax
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key)) # (bs, (dx, dy, dw, dh, score))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            # rnp_data = [rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights]
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
            rpn_label, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_label.view(batch_size, -1)
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))

            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            #self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            self.rpn_loss_cls = smooth_classification_loss(cls_score=rpn_cls_score, label=rpn_label,
                                                           smooth=cfg.TRAIN.CLASSIFICATION_SMOOTH_LOSS)

            fg_cnt = torch.sum(rpn_label.data.ne(0))

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])
        return rois, self.rpn_loss_cls, self.rpn_loss_box
