
"""

"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.psroi_pooling.modules.psroi_pool import PSRoIPool
from lib.model.rpn.proposal_target_layer_cascade_gt_index import _ProposalTargetLayer
from lib.model.rpn.rpn import _RPN
from lib.model.utils.config import cfg
from lib.model.utils.net_utils import _smooth_l1_loss
from torch.autograd import Variable
import pdb


class RFCN_head(nn.Module):
    """ R-FCN """
    def __init__(self, classes, class_agnostic):
        super(RFCN_head, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.box_num_classes = 1 if class_agnostic else self.n_classes

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_psroi_pool_cls = PSRoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE,
                                          spatial_scale=1/16.0, group_size=cfg.POOLING_SIZE,
                                          output_dim=self.n_classes)
        self.RCNN_psroi_pool_loc = PSRoIPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE,
                                          spatial_scale=1/16.0, group_size=cfg.POOLING_SIZE,
                                          output_dim=self.box_num_classes * 4)
        self.pooling = nn.AvgPool2d(kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE


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

    def detect_loss(self, cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws):

        # bounding box regression L1 loss
        RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        # classification loss
        RCNN_loss_cls = F.cross_entropy(cls_score, rois_label) # cls_score: [N, 2], rois_label: [N]

        return RCNN_loss_cls, RCNN_loss_bbox

    def ohem_detect_loss(self, cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws):

        def log_sum_exp(x):
            x_max = x.data.max()
            return torch.log(torch.sum(torch.exp(x - x_max), dim=1, keepdim=True)) + x_max

        num_hard = cfg.TRAIN.BATCH_SIZE * self.batch_size
        pos_idx = rois_label > 0
        num_pos = pos_idx.int().sum()

        # classification loss
        num_classes = cls_score.size(1)
        weight = cls_score.data.new(num_classes).fill_(1.)
        weight[0] = num_pos.data[0] / num_hard

        conf_p = cls_score.detach()
        conf_t = rois_label.detach()

        # rank on cross_entropy loss
        loss_c = log_sum_exp(conf_p) - conf_p.gather(1, conf_t.view(-1,1))
        loss_c[pos_idx] = 100. # include all positive samples
        _, topk_idx = torch.topk(loss_c.view(-1), num_hard)
        loss_cls = F.cross_entropy(cls_score[topk_idx], rois_label[topk_idx], weight=weight)

        # bounding box regression L1 loss
        pos_idx = pos_idx.unsqueeze(1).expand_as(bbox_pred)
        loc_p = bbox_pred[pos_idx].view(-1, 4)
        loc_t = rois_target[pos_idx].view(-1, 4)
        loss_box = F.smooth_l1_loss(loc_p, loc_t)

        return loss_cls, loss_box

if __name__ == '__main__':
    print('runing rfcn_head...')
