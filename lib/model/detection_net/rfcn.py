from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
We get the v2 version of R-FCN to support the input proposals.
i.e. we use the boxes in trajectories to improve the detection 
performance (tracking performance). 
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
from lib.model.detection_net.rfcn_head import RFCN_head
from lib.model.utils.net_utils import _deltas_and_proposals_to_bboxes, _repulsion_loss
from lib.model.detection_net.resnet_atrous import resnet18, resnet34, resnet50, resnet101, resnet152
from lib.utils.visualization import show_feature_map, show_compressed_frame

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


class RFCN(RFCN_head):
    def __init__(self, classes, base_net='resnet', num_layers_i=101, pretrained=False,
                 class_agnostic=False):
        """
        :param classes: list, the classes name
        :param base_net: the base network for rfcn
        :param num_layers_i: the number of layers of resnet used for i frame
        :param pretrained: whether to use the pretrained resnet
        :param class_agnostic: the class_agnostic box regression for rpn
        """

        self.test_config = None  # used to preserve the configure for testing
        self.train_config = None  # used to preserve the configure for training

        self.base_net = base_net
        self.num_layers_i = num_layers_i
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        RFCN_head.__init__(self, classes, class_agnostic)

        if cfg.TRAIN.AMBIGUOUS_CLASS_ID is not None:
            if cfg.TRAIN.AMBIGUOUS_CLASS_ID in list(range(len(self.classes))):
                raise ValueError('The ambiguous class id should be a positive value,\n'
                                 'and it should be larger or equal to num_of_classes({}),\n'
                                 ' but get {}'.format(len(self.classes), cfg.TRAIN.AMBIGUOUS_CLASS_ID))

    def _init_modules(self):
        # ----------- build base cnn for feature extraction ---------------
        # resnet_i = eval('resnet{}()'.format(self.num_layers_i))
        # resnet_p = eval('resnet{}()'.format(self.num_layers_p))
        resnet_i = eval(self.base_net + '{}()'.format(self.num_layers_i))
        if self.pretrained:
            resnet_i_weight = model_zoo.load_url(model_urls['resnet{}'.format(self.num_layers_i)])
            load_pretrained_resnet_weights_to_our_modified_resnet(resnet_i, resnet_i_weight)
            print('load pretrained resnet{}'.format(self.num_layers_i) + ' model from Pytorch model zoo...')

        # Build resnet for I frame.
        # it should noted that the output of layer1,2,3,4 of resnet are not activated (before ReLu),
        # hence the output of cnn_base_i are not activated
        rcnn_base_i = [resnet_i.conv1,
                       resnet_i.bn1,
                       resnet_i.relu,
                       resnet_i.maxpool,
                       resnet_i.layer1,
                       resnet_i.relu,
                       resnet_i.layer2,
                       resnet_i.relu,
                       resnet_i.layer3]
        self.RCNN_base_i = nn.Sequential(*rcnn_base_i)

        # -------------- build another sub-module for psroipooling for calssification and box reression ---------
        self.RCNN_conv_new = nn.Sequential(
            resnet_i.layer4,
            resnet_i.relu,
            nn.Conv2d(in_channels=2048, out_channels=1024,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )

        # -------------- build box regression conv layer -------------------
        if self.class_agnostic:
            self.RCNN_bbox_base = nn.Conv2d(in_channels=1024, out_channels=4 * cfg.POOLING_SIZE * cfg.POOLING_SIZE,
                                            kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.RCNN_bbox_base = nn.Conv2d(in_channels=1024,
                                            out_channels=4 * self.n_classes * cfg.POOLING_SIZE * cfg.POOLING_SIZE,
                                            kernel_size=1, stride=1, padding=0, bias=False)

        # --------------- build classification conv layer -------------------
        self.RCNN_cls_base = nn.Conv2d(in_channels=1024,
                                       out_channels=self.n_classes * cfg.POOLING_SIZE * cfg.POOLING_SIZE,
                                       kernel_size=1, stride=1, padding=0, bias=False)

        # Fix blocks for RCNN_base_i
        for p in self.RCNN_base_i[0].parameters(): p.requires_grad = False
        for p in self.RCNN_base_i[1].parameters(): p.requires_grad = False # bn layer

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_base_i[8].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_base_i[6].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_base_i[4].parameters(): p.requires_grad = False

        # Fix some layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.RCNN_base_i.apply(set_bn_fix)
        self.RCNN_conv_new.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base_i.eval()

            index = [8, 6, 4]
            index = index[0:3-cfg.RESNET.FIXED_BLOCKS]

            for layer in index:
            #for fix_layer in range(8, 3 + cfg.RESNET.FIXED_BLOCKS, -1):
                self.RCNN_base_i[layer].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base_i.apply(set_bn_eval)
            self.RCNN_conv_new.apply(set_bn_eval)

    def set_train_and_test_configure(self, phase, config=None):
        """
        This function set the modul for
        :param phase: 'train' or 'test'
        :param config: 'rcnn_base_i' or 'rcnn_base_p' or 'joint'
        :return:
        """
        if phase == 'test':
            self.train(False)
            self.test_config = config
        elif phase == 'train':
            self.train(True)
            self.train_config = config
        else:
            raise RuntimeError('Unrecognized phase: ' + phase)

    """ ------------------------------------------------------------------------------"""
    """-------------------- def some forward functions -------------------------------"""
    """-------------------------------------------------------------------------------"""

    def _forward_train_rcnn_base_i(self, im_info, i_frame, i_frame_box, i_num_box):
        """
        This function train the rcnn_base_i branch and the headers. The following data are all Variables
        :param im_info: 2D tensor, bs x 3, [h, w, scale]
        :param i_frame: 4D tensor, bs x 8 x h x w, the 0:3 channel is image, 3:5 channel is motion vector,
                    5:8 channel is the residual
        :param i_frame_box: 2D tensor, bs x num_box x 5, each row is (x1, y1, x2, y2, class_id)
        :param i_num_box: 2D tensor, bs x 1, the number of gt boxes of i frame
        """
        # # # -----------------------------------------------------
        # # show the resized image to make sure that we have cropped correctly
        # import numpy as np
        # mean_pixel = np.array([[[102.9801, 115.9465, 122.7717]]])
        # batch_size = i_frame.size(0)
        # idx = 0
        #
        # one_i_im = np.array(i_frame[idx, 0:3,:,:].permute(1, 2, 0).data.cpu().numpy() + mean_pixel, dtype=np.uint8)
        # show_compressed_frame(one_i_im, frame_type=0, save=True, title='i frame', show=False,
        #                       path='/home/liuqk/Program/pycharm/R-FCN-pytorch_v2/images/i_frame.pdf')
        # # ------------------------- end of show -----------------------------------------
        # we set a trace, if the weight of this layer has nan, then it will pause:
        # we use the fact that nan != nan is true
        rcnn_conv_new_2_weight = self.RCNN_conv_new.state_dict()['2.weight']
        if (rcnn_conv_new_2_weight != rcnn_conv_new_2_weight).sum() > 0:
            raise RuntimeError('\n there is nan in the weight of one layer\n')

        relu = nn.ReLU(inplace=True)
        batch_size = i_frame.size(0)
        self.batch_size = batch_size

        im_info = im_info.data
        im_info = im_info[:, 0:3]  # we do not need the frame type in this stage
        i_frame_box = i_frame_box.data[:, :, 0:5].contiguous() # size [bs, num_box, 5], (x1, y1, x2, y2, class_id)
        i_num_box = i_num_box.data
        if i_frame.size()[1] == 3: # the input image data only has image
            base_feat_before_relu = self.RCNN_base_i(i_frame)
        else: # the input data may also has mv and residual
            base_feat_before_relu = self.RCNN_base_i(i_frame[:, 0:3, :, :])

        base_feat_out = base_feat_before_relu.clone()  # used to output as appearance feature, [bs, c, h, w]
        f_h, f_w = base_feat_out.size()[2], base_feat_out.size()[3]
        im_h, im_w = i_frame.size()[2], i_frame.size()[3]
        h_scale, w_scale = f_h / im_h, f_w / im_w
        f_scale = i_frame_box.new([w_scale, h_scale, w_scale, h_scale])  # [4]

        base_feat = relu(base_feat_before_relu)

        # feat base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, i_frame_box, i_num_box)

        # proposal
        roi_data = self.RCNN_proposal_target(rois, i_frame_box, i_num_box)
        rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, roi_gt_index = roi_data
        # rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
        # the rois has the size [bs, num_rois, 5], and the [:, :, 0] is the batch index

        rois_label = Variable(rois_label.view(-1).long())
        rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

        # regression and classification
        rois = Variable(rois)
        base_feat = self.RCNN_conv_new(base_feat)

        cls_feat = self.RCNN_cls_base(base_feat)
        pooled_feat_cls = self.RCNN_psroi_pool_cls(cls_feat, rois.view(-1, 5))
        cls_score = self.pooling(pooled_feat_cls)
        cls_score = cls_score.squeeze()

        bbox_base = self.RCNN_bbox_base(base_feat)
        pooled_feat_loc = self.RCNN_psroi_pool_loc(bbox_base, rois.view(-1, 5))
        bbox_pred = self.pooling(pooled_feat_loc)
        bbox_pred = bbox_pred.squeeze()

        # select the corresponding columns according to roi labels (remove the bg deltas)
        if not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            #print(bbox_pred_view)
            #print(rois_label)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute regression and classification loss
        loss_func = self.ohem_detect_loss if cfg.TRAIN.OHEM else self.detect_loss
        RCNN_loss_cls, RCNN_loss_bbox = loss_func(cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = F.softmax(cls_score, dim=1)
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        if self.train_config == 'rcnn_base_i':
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
        elif self.train_config == 'joint_with_sbc': #
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, base_feat_out, f_scale

    def _forward_test_rcnn_base_i(self, im_info, i_frame, i_frame_box, i_num_box, pre_boxes=None):
        """
        This function test the performance rcnn_base_i branch and the headers. The
        following data are all Variables. Currently, this function only support 1 GPU

        TODO: support test with multi-GPUs

        :param im_info: 2D tensor, bs x (3 + num_of_frame_one_batch)
        :param i_frame: 4D tensor, bs x 3 x h x w, the 0:3 channel is image
        :param i_frame_box: 3D tensor, bs x num_box x 5, each row is (x1, y1, x2, y2, class_id). In test phase,
                    it has the size bs x 1 x 5, and all values are set to 1
        :param i_num_box: 2D tensor, bs x 1, the number of gt boxes of i frame, in test phase, all values are
                    set to 0
        :param pre_boxes: 3D tensor, bs x num_pre_box x 5, the boxes in pre-frame
        """
        relu = nn.ReLU(inplace=True)

        batch_size = i_frame.size(0)
        self.batch_size = batch_size

        im_info = im_info.data
        im_info = im_info[:, 0:3]  # in this stage, we do not need the frame type
        i_frame_box = i_frame_box.data[:, :, 0:5].contiguous() # size [bs, num_box, 5], [x1, y1 ,x2, y2, class_id]
        i_num_box = i_num_box.data
        if i_frame.size()[1] == 3:
            base_feat_before_relu = self.RCNN_base_i(i_frame)
        else:
            base_feat_before_relu = self.RCNN_base_i(i_frame[:, 0:3, :, :])

        # the
        base_feat_out = base_feat_before_relu.clone()  # used to output as appearance feature, [bs, c, h, w]
        f_h, f_w = base_feat_out.size()[2], base_feat_out.size()[3]
        im_h, im_w = i_frame.size()[2], i_frame.size()[3]
        h_scale, w_scale = f_h / im_h, f_w / im_w
        f_scale = i_frame_box.new([w_scale, h_scale, w_scale, h_scale])  # [4]

        base_feat = relu(base_feat_before_relu)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, i_frame_box, i_num_box)

        rois = Variable(rois)
        if pre_boxes is not None:
            rois = torch.cat((rois, pre_boxes), dim=1)

        base_feat = self.RCNN_conv_new(base_feat)

        # do roi pooling based predicted rois
        cls_feat = self.RCNN_cls_base(base_feat)
        pooled_feat_cls = self.RCNN_psroi_pool_cls(cls_feat, rois.view(-1, 5))
        cls_score = self.pooling(pooled_feat_cls)
        cls_score = cls_score.view(cls_score.size(0), cls_score.size(1))
        cls_prob = F.softmax(cls_score, dim=1)

        bbox_base = self.RCNN_bbox_base(base_feat)
        pooled_feat_loc = self.RCNN_psroi_pool_loc(bbox_base, rois.view(-1, 5))
        bbox_pred = self.pooling(pooled_feat_loc)
        bbox_pred = bbox_pred.squeeze()

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, base_feat_out, f_scale

    def _forward_extract_base_features(self, i_frame, pre_boxes=None):
        """
        This function test the performance rcnn_base_i branch and the headers. The
        following data are all Variables. Currently, this function only support 1 GPU

        :param i_frame: 4D tensor, bs x 3 x h x w, the 0:3 channel is image
        :param pre_boxes: 3D tensor, bs x num_pre_box x 5, the boxes in pre-frame
        """
        relu = nn.ReLU(inplace=True)

        batch_size = i_frame.size(0)
        self.batch_size = batch_size

        if i_frame.size()[1] == 3:
            base_feat_before_relu = self.RCNN_base_i(i_frame)
        else:
            base_feat_before_relu = self.RCNN_base_i(i_frame[:, 0:3, :, :])

        # the
        base_feat_out = base_feat_before_relu.clone()  # used to output as appearance feature, [bs, c, h, w]
        f_h, f_w = base_feat_out.size()[2], base_feat_out.size()[3]
        im_h, im_w = i_frame.size()[2], i_frame.size()[3]
        h_scale, w_scale = f_h / im_h, f_w / im_w
        f_scale = i_frame.data.new([w_scale, h_scale, w_scale, h_scale])  # [4]

        if pre_boxes is None:
            return base_feat_out, f_scale
        else:

            base_feat = relu(base_feat_before_relu)

            rois = pre_boxes

            base_feat = self.RCNN_conv_new(base_feat)

            # do roi pooling based predicted rois
            cls_feat = self.RCNN_cls_base(base_feat)
            pooled_feat_cls = self.RCNN_psroi_pool_cls(cls_feat, rois.view(-1, 5))
            cls_score = self.pooling(pooled_feat_cls)
            cls_score = cls_score.view(cls_score.size(0), cls_score.size(1))
            cls_prob = F.softmax(cls_score, dim=1)

            bbox_base = self.RCNN_bbox_base(base_feat)
            pooled_feat_loc = self.RCNN_psroi_pool_loc(bbox_base, rois.view(-1, 5))
            bbox_pred = self.pooling(pooled_feat_loc)
            bbox_pred = bbox_pred.squeeze()

            cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
            bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

            return rois, cls_prob, bbox_pred, base_feat_out, f_scale

    # def forward(self, im_info, i_frame, i_frame_box, i_num_box, p_frame=None, p_frame_box=None, p_num_box=None):
    def forward(self, *inputs):
        # im_info, i_frame, i_frame_box, i_num_box, p_frame, p_frame_box, p_num_box
        if self.training:
            if self.train_config is None:
                raise RuntimeError(
                    'please call function set_train_and_test_configure() before calling forward() of this model.')
            if self.train_config == 'rcnn_base_i' or self.train_config == 'joint_with_sbc':
                # in this stage, we just want train the rcnn_base_i and the headers,
                # hence the whole model can be treated as a RFCN model. The data we
                # need is I frame, i.e. the origin image.
                return self._forward_train_rcnn_base_i(*inputs)
            else:
                raise ValueError('unrecognized train module: {}'.format(self.train_config))

        else:  # in testing phase
            if self.test_config is None:
                raise ValueError(
                    'please call function set_train_and_test_configure() before calling forward() of this model.')

            if self.test_config == 'rcnn_base_i':
                # in this phase, we test the detection performance of rcnn_base_i and
                # the headers. And the whole model can be treated as a rfcn network
                return self._forward_test_rcnn_base_i(*inputs)
            if self.test_config == 'extract_base_features':
                return self._forward_extract_base_features(*inputs)
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

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_conv_new[2], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_base, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_base, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


if __name__ == '__main__':
    print('runing rfcn...')
