# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import numpy as np
import pdb
from torch.autograd import Variable
import torch.nn.functional as F
import math

# def bbox_transform(ex_rois, gt_rois):
#     ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
#     ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
#     ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
#     ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
#
#     gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
#     gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
#     gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
#     gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
#
#     targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
#     targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
#     targets_dw = torch.log(gt_widths / ex_widths)
#     targets_dh = torch.log(gt_heights / ex_heights)
#
#     targets = torch.stack(
#         (targets_dx, targets_dy, targets_dw, targets_dh),1)
#
#     return targets
#
# def bbox_transform_batch(ex_rois, gt_rois):
#
#     if ex_rois.dim() == 2:
#         ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
#         ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
#         ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
#         ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
#
#         gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
#         gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
#         gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
#         gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
#
#         targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
#         targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
#         targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
#         targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))
#
#     elif ex_rois.dim() == 3:
#         ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
#         ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
#         ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
#         ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights
#
#         gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
#         gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
#         gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
#         gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
#
#         targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
#         targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
#         targets_dw = torch.log(gt_widths / ex_widths)
#         targets_dh = torch.log(gt_heights / ex_heights)
#     else:
#         raise ValueError('ex_roi input dimension is not correct.')
#
#     targets = torch.stack(
#         (targets_dx, targets_dy, targets_dw, targets_dh),2)
#
#     return targets
#
# def bbox_transform_inv(boxes, deltas, batch_size):
#     widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
#     heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
#     ctr_x = boxes[:, :, 0] + 0.5 * widths
#     ctr_y = boxes[:, :, 1] + 0.5 * heights
#
#     dx = deltas[:, :, 0::4]
#     dy = deltas[:, :, 1::4]
#     dw = deltas[:, :, 2::4]
#     dh = deltas[:, :, 3::4]
#
#     pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
#     pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
#     pred_w = torch.exp(dw) * widths.unsqueeze(2)
#     pred_h = torch.exp(dh) * heights.unsqueeze(2)
#
#     pred_boxes = deltas.clone()
#     # x1
#     pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
#     # y1
#     pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
#     # x2
#     pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
#     # y2
#     pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h
#
#     return pred_boxes
#
# def bbox_transform_inv_one_class(boxes, deltas):
#     """
#     In order to make the loss backward, the variable can not be assigned in-place.
#     Hence, we implement this inverse transform function to perform the inversion
#     of transformation. But this function currently support the prediction of one
#     class only!
#     :param boxes:
#     :param deltas:
#     :return:
#     """
#     widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
#     heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
#     ctr_x = boxes[:, :, 0] + 0.5 * widths
#     ctr_y = boxes[:, :, 1] + 0.5 * heights
#
#     dx = deltas[:, :, 0]
#     dy = deltas[:, :, 1]
#     dw = deltas[:, :, 2]
#     dh = deltas[:, :, 3]
#
#     pred_ctr_x = dx * widths + ctr_x
#     pred_ctr_y = dy * heights + ctr_y
#     pred_w = torch.exp(dw) * widths
#     pred_h = torch.exp(dh) * heights
#
#
#     pred_x1 = pred_ctr_x - 0.5 * pred_w
#     pred_y1 = pred_ctr_y - 0.5 * pred_h
#     pred_x2 = pred_ctr_x + 0.5 * pred_w
#     pred_y2 = pred_ctr_y + 0.5 * pred_h
#
#     pred_boxes = torch.stack((pred_x1, pred_y1, pred_x2, pred_y2), dim=2)
#
#     return pred_boxes


def bbox_transform(ex_rois, gt_rois, sigma=1/math.sqrt(2)):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) * math.sqrt(2) * sigma / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) * math.sqrt(2) * sigma / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),1)

    return targets


def bbox_transform_batch(ex_rois, gt_rois, sigma=1/math.sqrt(2)):

    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        # targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) * 2 * sigma / ex_widths
        # targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) * 2 * sigma / ex_heights

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) * math.sqrt(2) * sigma / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) * math.sqrt(2) * sigma / ex_heights

        targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) * math.sqrt(2) * sigma / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) * math.sqrt(2) * sigma / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),2)

    return targets


def bbox_transform_inv(boxes, deltas, batch_size=None, sigma=1/math.sqrt(2)):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) / (math.sqrt(2) * sigma) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) / (math.sqrt(2) * sigma) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def bbox_transform_inv_one_class(boxes, deltas, sigma=1/math.sqrt(2)):
    """
    In order to make the loss backward, the variable can not be assigned in-place.
    Hence, we implement this inverse transform function to perform the inversion
    of transformation. But this function currently support the prediction of one
    class only!
    :param boxes:
    :param deltas:
    :return:
    """
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0]
    dy = deltas[:, :, 1]
    dw = deltas[:, :, 2]
    dh = deltas[:, :, 3]

    pred_ctr_x = dx * widths / (math.sqrt(2) * sigma) + ctr_x
    pred_ctr_y = dy * heights / (math.sqrt(2) * sigma) + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = torch.stack((pred_x1, pred_y1, pred_x2, pred_y2), dim=2)

    return pred_boxes


def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
    boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y

    return boxes

def clip_boxes(boxes, im_shape, batch_size):

    if not isinstance(boxes, Variable):
        for i in range(batch_size):
            boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1) # x1
            boxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1) # y1
            boxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1) # x2
            boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1) # y2
    else:
        for i in range(batch_size):
            xyxy = Variable(im_shape.data.new([0, 0, 0, 0]))
            xyxy = xyxy.unsqueeze(dim=0)
            xyxy = xyxy.repeat(int(boxes[i].size()[0]), int(boxes[i].size()[1]/4)) # [N, 4*A]

            boxes[i, :, 0::4], _ = torch.stack((boxes[i, :, 0::4], xyxy[:, 0::4]), 0).max(0)
            boxes[i, :, 1::4], _ = torch.stack((boxes[i, :, 1::4], xyxy[:, 1::4]), 0).max(0)
            boxes[i, :, 2::4], _ = torch.stack((boxes[i, :, 2::4], xyxy[:, 2::4]), 0).max(0)
            boxes[i, :, 3::4], _ = torch.stack((boxes[i, :, 3::4], xyxy[:, 3::4]), 0).max(0)

            xyxy = Variable(im_shape.data.new([im_shape[i, 1].data[0]-1, im_shape[i, 0].data[0]-1, im_shape[i, 1].data[0]-1, im_shape[i, 0].data[0]-1]))
            xyxy = xyxy.unsqueeze(dim=0)
            xyxy = xyxy.repeat(int(boxes[i].size()[0]), int(boxes[i].size()[1]/4)) # [N, 4*A]

            boxes[i, :, 0::4], _ = torch.stack((boxes[i, :, 0::4], xyxy[:, 0::4]), 0).min(0)
            boxes[i, :, 1::4], _ = torch.stack((boxes[i, :, 1::4], xyxy[:, 1::4]), 0).min(0)
            boxes[i, :, 2::4], _ = torch.stack((boxes[i, :, 2::4], xyxy[:, 2::4]), 0).min(0)
            boxes[i, :, 3::4], _ = torch.stack((boxes[i, :, 3::4], xyxy[:, 3::4]), 0).min(0)

    return boxes
    #for i in range(batch_size):
    #    boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1) # x1
   #       boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1) # y2
#
    #return boxes


def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)


    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:,:,:4].contiguous()


        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:,:,:4].contiguous()
        else:
            anchors = anchors[:,:,1:5].contiguous()

        gt_boxes = gt_boxes[:,:,:4].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)

        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps


def bbox_iou(boxes1, boxes2):
    """
    This function compute the iou between boxes1 and boxes2.
    :param boxes1: 2D tensor, [N, 4], each row is  [x1, y1, x2, y2]
    :param boxes2: 2D tensor, [M, 4], each row is  [x1, y1, x2, y2]
    :return: 2D tensor with size [N, M]
    """
    N, M = boxes1.size()[0], boxes2.size()[0]

    # expand boxes1 to size [N, M, 4], i.e. expand each box in boxes1 M times
    boxes1 = boxes1.unsqueeze(dim=1)
    boxes1 = boxes1.repeat(1, M, 1)

    # expand boxes2 to size [N, M, 4], i.e. expand each box in boxes2 N times
    boxes2 = boxes2.unsqueeze(dim=0)
    boxes2 = boxes2.repeat(N, 1, 1)

    # tl_1, br_1, tl_2, br_2 has the size [N, M, 2]
    tl_1, br_1 = boxes1[:, :, 0:2], boxes1[:, :, 2:4] # the coordinates of top-left, bottom-right
    tl_2, br_2 = boxes2[:, :, 0:2], boxes2[:, :, 2:4]

    tl = torch.stack((tl_1, tl_2), dim=0) # [x, N, M, 2]
    tl, _ = tl.max(dim=0) # [N, M ,2]

    br = torch.stack((br_1, br_2), dim=0)
    br, _ = br.min(dim=0)

    wh = br - tl # N x M x 2
    mask = wh < 0
    wh[mask] = 0

    area_intersection = torch.prod(wh, dim=2)
    area_1 = torch.prod(boxes1[:, :, 2:4] - boxes1[:, :, 0:2], dim=2)
    area_2 = torch.prod(boxes2[:, :, 2:4] - boxes2[:, :, 0:2], dim=2)

    return area_intersection / (area_1 + area_2 - area_intersection)

def bbox_iou_batch(boxes1, boxes2):
    """
    This function compute the iou between boxes1 and boxes2.
    :param boxes1: 3D tensor, [bs, N, 4], each row is  [x1, y1, x2, y2]
    :param boxes2: 2D tensor, [bs, M, 4], each row is  [x1, y1, x2, y2]
    :return: 2D tensor with size [bs, N, M]
    """
    bs = boxes1.size()[0]
    N, M = boxes1.size()[1], boxes2.size()[1]
    # iou = torch.FloatTensor(bs, N, M).fill_(0)
    # if boxes1.is_cuda:
    #     iou = iou.cuda()
    #
    # for i in range(bs):
    #     iou[i] = bbox_iou(boxes1[i], boxes2[i])
    #
    # return iou


    # expand boxes1 to size [bs, N, M, 4], i.e. expand each box in boxes1 M times
    boxes1 = boxes1.unsqueeze(dim=2) # [bs, N, 1, 4]
    boxes1 = boxes1.repeat(1, 1, M, 1) # [bs, N, M, 4]

    # expand boxes2 to size [N, M, 4], i.e. expand each box in boxes2 N times
    boxes2 = boxes2.unsqueeze(dim=1) # [bs, 1, M, 4]
    boxes2 = boxes2.repeat(1, N, 1, 1) # [bs, N, M, 4]

    tl_1, br_1 = boxes1[:, :, :, 0:2], boxes1[:, :, :, 2:4] # the coordinates of top-left, bottom-right
    tl_2, br_2 = boxes2[:, :, :, 0:2], boxes2[:, :, :, 2:4]

    tl = torch.stack((tl_1, tl_2), dim=0)
    tl, _ = torch.max(tl, dim=0)

    br = torch.stack((br_1, br_2), dim=0)
    br, _ = torch.min(br, dim=0)

    wh = br - tl # bs x N x M x 2
    mask = wh < 0
    wh[mask] = 0

    area_intersection = torch.prod(wh, dim=3) # [bs, N, M]
    area_1 = torch.prod(boxes1[:, :, :, 2:4] - boxes1[:, :, :, 0:2], dim=3)
    area_2 = torch.prod(boxes2[:, :, :, 2:4] - boxes2[:, :, :, 0:2], dim=3)

    return area_intersection / (area_1 + area_2 - area_intersection)
















