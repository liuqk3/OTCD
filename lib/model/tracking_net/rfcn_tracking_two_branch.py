from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Tracking net. There are two branches in the model. One is for motion vector,
the other is for residual. Then the feature obtained from two branches are 
concated together.
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import pdb
from torch.autograd import Variable
from lib.model.utils.config import cfg
from lib.utils.misc import load_pretrained_resnet_weights_to_our_modified_resnet
import lib.utils.misc as misc
from lib.model.roi_align.roi_align.roi_align import RoIAlign
from lib.model.tracking_net.rfcn_tracking_head import RFCN_tracking_head
from lib.model.rpn.bbox_transform import bbox_transform_batch
from lib.model.utils.net_utils import _smooth_l1_loss
from lib.model.detection_net.resnet_atrous import resnet18, resnet34, resnet50, resnet101, resnet152
from lib.utils.visualization import show_feature_map, show_compressed_frame

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


class RFCN_tracking(RFCN_tracking_head):
    def __init__(self, base_net='resnet', num_layers_mv=34, num_layers_residual=34, pretrained=False,
                 transform_sigma=0.5, pooling_size=7):
        """
        :param base_net: the base network for rfcn
        :param num_layers_mv: the number of layers of resnet used for i frame
        :param num_layers_residual:  the number of layers of resnet used for residual
        :param pretrained: whether to use the pretrained resnet
        :param transform_sigma: scalar, used for box transformation and inv-transformation.
        :param pooling_size: scalar, the size of PSROIPooling
        """
        # TODO support different layers for motion vector and resnet
        self.test_config = None  # used to preserve the configure for testing
        self.train_config = None  # used to preserve the configure for training

        # define some attributes for tracking
        self.accumulated = None  # used to denote the motion vector and residual are accumulated

        self.base_net = base_net
        self.num_layers_mv = num_layers_mv
        self.num_layers_residual = num_layers_residual

        if self.num_layers_mv != self.num_layers_residual:
            raise ValueError('The layers of base net for mv {} and residual {} are not the same!'.
                             format(self.num_layers_mv, num_layers_residual))

        self.pretrained = pretrained
        self.transform_sigma = transform_sigma

        self.pooling_size = pooling_size

        RFCN_tracking_head.__init__(self, pooling_size=self.pooling_size)


    def _init_modules(self):
        # ----------- build base cnn for feature extraction ---------------
        resnet_mv = eval(self.base_net + '{}()'.format(self.num_layers_mv))
        resnet_residual = eval(self.base_net + '{}()'.format(self.num_layers_residual))
        if self.pretrained:
            resnet_mv_weight = model_zoo.load_url(model_urls['resnet{}'.format(self.num_layers_mv)])
            load_pretrained_resnet_weights_to_our_modified_resnet(resnet_mv, resnet_mv_weight)
            print('load pretrained resnet{}'.format(self.num_layers_mv) + ' model from Pytorch model zoo...')

            resnet_residual_weight = model_zoo.load_url(model_urls['resnet{}'.format(self.num_layers_residual)])
            load_pretrained_resnet_weights_to_our_modified_resnet(resnet_residual, resnet_residual_weight)
            print('load pretrained resnet{}'.format(self.num_layers_residual) + ' model from Pytorch model zoo...')

        # Build resnet for motion vector.
        rcnn_base_mv = [#resnet_mv.conv1,
                        nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
                        resnet_mv.bn1,
                        resnet_mv.relu,
                        resnet_mv.maxpool,
                        resnet_mv.layer1,
                        resnet_mv.relu,
                        resnet_mv.layer2,
                        resnet_mv.relu,
                        resnet_mv.layer3,
                        resnet_mv.relu,
                        resnet_mv.layer4,
                        resnet_mv.relu]
        self.RCNN_base_mv = nn.Sequential(*rcnn_base_mv)

        # Build resnet for residual
        rcnn_base_residual = [resnet_residual.conv1,
                              resnet_residual.bn1,
                              resnet_residual.relu,
                              resnet_residual.maxpool,
                              resnet_residual.layer1,
                              resnet_residual.relu,
                              resnet_residual.layer2,
                              resnet_residual.relu,
                              resnet_residual.layer3,
                              resnet_residual.relu,
                              resnet_residual.layer4,
                              resnet_residual.relu]

        self.RCNN_base_residual = nn.Sequential(*rcnn_base_residual)

        # check the output channel of the cnn_base_mv and cnn_base_residual
        num_channel_mv = 2048
        if self.num_layers_mv == 18 or self.num_layers_mv == 34:
            num_channel_mv = 512

        num_channel_residual = 512
        if self.num_layers_residual == 50 or self.num_layers_residual == 101 or self.num_layers_residual == 152:
            num_channel_residual = 2048

        # -------------------- build a conv layer to add residual and mv ---------------------------
        # we first concat the features from mv and residual, and then
        # forward through the following conv layer
        self.feature_add_conv = nn.Conv2d(in_channels=num_channel_mv + num_channel_residual,
                                          out_channels=1024,
                                          kernel_size=1, stride=1, padding=0, bias=False)

        # -------------- build box regression conv layer -------------------
        self.RCNN_bbox_base = nn.Conv2d(in_channels=1024, out_channels=4*self.pooling_size*self.pooling_size,
                                            kernel_size=1, stride=1, padding=0, bias=False)

        # Fix some layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.RCNN_base_mv.apply(set_bn_fix)
        self.RCNN_base_residual.apply(set_bn_fix)

        assert (0 <= cfg.RESNET.FIXED_BLOCKS <= 4) # set this value to 0, so we can train all blocks
        if cfg.RESNET.FIXED_BLOCKS >= 4: # fix all blocks
            for p in self.RCNN_base_mv[10].parameters(): p.requires_grad = False
            for p in self.RCNN_base_residual[10].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 3: # fix first 3 blocks
            for p in self.RCNN_base_mv[8].parameters(): p.requires_grad = False
            for p in self.RCNN_base_residual[8].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2: # fix first 2 blocks
            for p in self.RCNN_base_mv[6].parameters(): p.requires_grad = False
            for p in self.RCNN_base_residual[6].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1: # fix first 1 block
            for p in self.RCNN_base_mv[4].parameters(): p.requires_grad = False
            for p in self.RCNN_base_residual[4].parameters(): p.requires_grad = False

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base_mv.eval()
            self.RCNN_base_residual.eval()

            index = [10, 8, 6, 4]
            index = index[0:4-cfg.RESNET.FIXED_BLOCKS]

            for layer in index:
                self.RCNN_base_mv[layer].train()
                self.RCNN_base_residual[layer].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base_mv.apply(set_bn_eval)
            self.RCNN_base_residual.apply(set_bn_eval)

    def set_train_and_test_configure(self, phase, config, accumulated=None):
        """
        This function set the modul for
        :param phase: 'train' or 'test'
        :param config: 'rcnn_base_i' or 'rcnn_base_p' or 'joint'
        :param accumulated: bool, used to denote the motion vector and
                    residual are accumulated
        :return:
        """
        if phase == 'test':
            self.train(False)
            self.test_config = config
            self.accumulated = accumulated
        elif phase == 'train':
            self.train(True)
            self.train_config = config
            if config == 'rcnn_base_residual':  # train rcnn_base_p only
                for p in self.RCNN_base_mv.parameters(): p.requires_grad = False
            elif config == 'rcnn_base_mv':  # train rcnn_base_i only
                for p in self.RCNN_base_residual.parameters(): p.requires_grad = False
            elif config == 'joint':
                # train the whole network, so we do nothing
                pass
            else:
                raise RuntimeError('Unrecognized module: ' + config)
        else:
            raise RuntimeError('Unrecognized phase: ' + phase)

    def _compute_bbox_targets(self, ex_rois, target_rois):
        """Compute bounding-box regression targets for an image.
        :param ex_rois: the extracted rois, 3D tensor, [bs, num_box, 4]
        :param target_rois: the rois that wanted to be, 3D tensor, [bs, num_box, 4]
        :param sigma: scalar, only used for our own targets computation.
        """

        assert ex_rois.size(1) == target_rois.size(1)
        assert ex_rois.size(2) == 4
        assert target_rois.size(2) == 4

        targets = bbox_transform_batch(ex_rois, target_rois, sigma=self.transform_sigma) # [bs, num_box, 4]

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).expand_as(targets))
                       / targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).expand_as(targets))
        return targets


    """ ------------------------------------------------------------------------------"""
    """-------------------- def some forward functions -------------------------------"""
    """-------------------------------------------------------------------------------"""

    def _forward_train_joint(self, frame_1_box, frame_2, frame_2_box, num_box):
        """
        This function use the rcnn_base_mv branch and the rcnn_base_residual. The following data are all Variables.
        We are trying to predict the offset between the boxes in frame 1 and frame 2 (i.e. frame_1_box and
        frame_2_box). We crop the feature using PSRoIPooling based frame_1_box to predict the offsets.

        :param frame_1_box: 3D tensor, bs x num_box x 6, each row is (x1, y1, x2, y2, class_id, target_id)
        :param frame_2: 4D tensor, bs x 5 x h x w, the 0:2 channel is motion vector, 2:5 channel is the residual
        :param frame_2_box: 2D tensor, bs x num_box x 6, each row is (x1, y1, x2, y2, class_id, target_id)
        :param num_box: 1D tensor, [bs], the number of gt boxes in differernt frames. Noted that the boxes in
            two frames that in one pair are the same.

        """
        # we set a trace, if the weight of this layer has nan, then it will pause:
        # we use the fact that nan != nan is true
        feature_add_conv_weight = self.feature_add_conv.state_dict()['weight']
        if (feature_add_conv_weight != feature_add_conv_weight).sum() > 0:
            print('\n there is nan in the weight of one layer\n')
            pdb.set_trace()

        batch_size = frame_2.size()[0]

        # get the base features
        feat_mv = self.RCNN_base_mv(frame_2[:, 0:2, :, :].contiguous())
        feat_residual = self.RCNN_base_residual(frame_2[:, 2:5].contiguous())

        # concate the features
        base_feat = torch.cat((feat_mv, feat_residual), dim=1)
        base_feat = self.feature_add_conv(base_feat)
        base_feat_loc = self.RCNN_bbox_base(base_feat)

        # PSRoIPooling
        frame_1_box_tmp = frame_1_box.data.contiguous() # [bs, num_box, 6], each row is [x1, y1, x2, y2, class_id, target_id]
        frame_2_box_tmp = frame_2_box.data.contiguous() # [bs, num_box, 6]

        # (1) generate rois
        rois_1 = frame_1_box_tmp.new(batch_size, frame_1_box_tmp.size()[1], 5).zero_() # each row is [batch_index, x1, y1, x2, y2]
        rois_1[:, :, 1:5] = frame_1_box_tmp[:, :, 0:4].clone()
        for bs_idx in range(batch_size):
            rois_1[bs_idx, :, 0] = bs_idx
        rois_1 = Variable(rois_1)

        # (2) pooling to get the offset
        pooled_feat_loc = self.RCNN_psroi_pool_loc(base_feat_loc, rois_1.view(-1, 5)) # [num_box, 4, pooled_size, pooled_size]
        bbox_pred = self.pooling(pooled_feat_loc) # [num_box, 4, 1, 1]
        bbox_pred = bbox_pred.squeeze() # [num_box, 4]

        bbox_pred = bbox_pred.view(batch_size, -1, 4)

        # compute the box regression target
        rois_1 = frame_1_box_tmp[:, :, 0:4].clone().contiguous()
        rois_2 = frame_2_box_tmp[:, :, 0:4].clone().contiguous()

        regression_targets = self._compute_bbox_targets(rois_1, rois_2)

        # compute the inside weights and outside weights
        num_box_1_tmp = num_box.data.int()
        # if (num_box_1_tmp == 0).sum() > 0:
        #     a = 1

        inside_weight = regression_targets.new(batch_size, regression_targets.size(1), 4).zero_()
        outside_weight = regression_targets.new(batch_size, regression_targets.size(1), 4).zero_()
        for bs_idx in range(batch_size):
            if num_box_1_tmp[bs_idx] > 0:
                inside_weight[bs_idx, 0:num_box_1_tmp[bs_idx], :] = 1
                outside_weight[bs_idx, 0:num_box_1_tmp[bs_idx], :] = 1.0 / num_box_1_tmp[bs_idx]

        # get the loss
        regression_targets = Variable(regression_targets)
        inside_weight = Variable(inside_weight)
        outside_weight = Variable(outside_weight)
        loss_bbox = _smooth_l1_loss(bbox_pred, regression_targets, inside_weight, outside_weight, dim=[2, 1])


        # #
        # outside_weight = outside_weight.data
        # outside_weight[outside_weight > 0] = 1
        # outside_weight = Variable(outside_weight)
        # bbox_pred = bbox_pred.view(-1, 4)
        # regression_targets = regression_targets.view(-1, 4)
        # inside_weight = inside_weight.view(-1, 4)
        # outside_weight = outside_weight.view(-1, 4)
        # loss_bbox_1 = _smooth_l1_loss(bbox_pred, regression_targets, inside_weight, outside_weight)

        return bbox_pred, loss_bbox

    def _forward_test_joint(self, frame_1_box, frame_2):
        """
           This function use the rcnn_base_mv branch and the rcnn_base_residual. The following data are all Variables.
           We are trying to predict the offset between the boxes in frame 1 and frame 2 (i.e. frame_1_box and
           frame_2_box). We crop the feature using PSRoIPooling based frame_1_box to predict the offsets.

           :param frame_1_box: 3D tensor, bs x num_box x 6, each row is (x1, y1, x2, y2, class_id, target_id)
           :param frame_2: 4D tensor, bs x 5 x h x w, the 0:2 channel is motion vector, 2:5 channel is the residual

        """
        # we set a trace, if the weight of this layer has nan, then it will pause:
        # we use the fact that nan != nan is true
        feature_add_conv_weight = self.feature_add_conv.state_dict()['weight']
        if (feature_add_conv_weight != feature_add_conv_weight).sum() > 0:
            print('\n there is nan in the weight of one layer\n')
            pdb.set_trace()

        batch_size = frame_2.size()[0]

        # get the base features
        feat_mv = self.RCNN_base_mv(frame_2[:, 0:2, :, :].contiguous())
        feat_residual = self.RCNN_base_residual(frame_2[:, 2:5].contiguous())

        # concate the features
        base_feat = torch.cat((feat_mv, feat_residual), dim=1)
        base_feat = self.feature_add_conv(base_feat)
        base_feat_loc = self.RCNN_bbox_base(base_feat)

        # PSRoIPooling
        frame_1_box_tmp = frame_1_box.data.contiguous()  # [bs, num_box, 6], each row is [x1, y1, x2, y2, class_id, target_id]

        # (1) generate rois
        rois_1 = frame_1_box_tmp.new(batch_size, frame_1_box_tmp.size()[1],
                                     5).zero_()  # each row is [batch_index, x1, y1, x2, y2]
        rois_1[:, :, 1:5] = frame_1_box_tmp[:, :, 0:4].clone()
        for bs_idx in range(batch_size):
            rois_1[bs_idx, :, 0] = bs_idx
        rois_1 = Variable(rois_1)

        # (2) pooling to get the offset
        pooled_feat_loc = self.RCNN_psroi_pool_loc(base_feat_loc,
                                                   rois_1.view(-1, 5))  # [num_box, 4, pooled_size, pooled_size]
        bbox_pred = self.pooling(pooled_feat_loc)  # [num_box, 4, 1, 1]
        bbox_pred = bbox_pred.squeeze()  # [num_box, 4]

        bbox_pred = bbox_pred.view(batch_size, -1, 4)

        return bbox_pred

    # def forward(self, im_info, i_frame, i_frame_box, i_num_box, p_frame=None, p_frame_box=None, p_num_box=None):
    def forward(self, *inputs):

        if self.training:
            if self.train_config is None:
                raise RuntimeError(
                    'please call function set_train_and_test_configure() before calling forward() of this model.')
            if self.train_config == 'joint':
                # in this stage, we use motion vector and residual together.
                return self._forward_train_joint(*inputs)

            else:
                raise ValueError('unrecognized train module: {}'.format(self.train_config))

        else:  # in testing phase
            if self.test_config is None:
                raise ValueError(
                    'please call function set_train_and_test_configure() before calling forward() of this model.')
            if self.test_config == 'joint':
                # in this stage, we use motion vector and residual together.
                return self._forward_test_joint(*inputs)
            else:
                raise ValueError('unrecognized test module: {}'.format(self.train_config))

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()

        # init the box regression conv layer
        normal_init(self.RCNN_bbox_base, 0, 0.001, cfg.TRAIN.TRUNCATED)
        # init the add_conv layer
        normal_init(self.feature_add_conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # init the first conv layer of rcnn_base_mv
        normal_init(self.RCNN_base_mv[0], 0, 0.01, cfg.TRAIN.TRUNCATED)

        if len(self.RCNN_base_residual) == 10:  # we add a conv layer to change the number of channels
            normal_init(self.RCNN_base_residual[9], 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


if __name__ == '__main__':
    print('runing rfcn...')
