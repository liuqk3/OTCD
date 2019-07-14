import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
from lib.model.utils.config import cfg
from lib.model.roi_crop.functions.roi_crop import RoICropFunction
import cv2
import pdb
import random
from lib.model.rpn.bbox_transform import bbox_transform_inv, bbox_transform_inv_one_class
from lib.model.rpn.bbox_transform import clip_boxes

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)

def vis_detections(im, class_name, dets, thresh=0.8,):
    """Visual debugging of detections.
    input:
        im: 3D array, H x W x 3, the image, uint8
        class_name: string
        dets: 2D array, N x (x1, y1, x2, y2, score)
    """
    for i in range(dets.shape[0]):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def save_checkpoint(state, filename):
    torch.save(state, filename)

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)

    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)

    loss_box = loss_box.mean()
    return loss_box


def smooth_classification_loss(cls_score, label, smooth=False, epsilon=0.1):
    """
    This function get the loss for the task of classification. We use the binary cross-entropy loss.
    :param cls_score: array_like Variable, [bs, num_class], the score before softmax
    :param label: [bs], the groundtruth label of the samples
    :param smooth: bool, whether to use the smoothed binary cross entropy
    :return: Variable, the loss
    """

    if not smooth:
        loss = F.cross_entropy(cls_score, label)
    elif smooth:
        if isinstance(epsilon, Variable):
            epsilon = epsilon.data[0]

        # get the batch size and number of classes
        batch_size, num_classes = cls_score.size()[0], cls_score.size()[1]

        # create the smooth label
        smooth_label = cls_score.data.new(batch_size, num_classes).fill_(0)

        label_expand = label.unsqueeze(dim=1).repeat(1, num_classes).data # [bs, num_classes]
        class_id = label.data.new(range(num_classes)).unsqueeze(dim=0) # [1, num_classes]
        class_id = class_id.repeat(batch_size, 1) # [bs, num_classes]
        #class_id = Variable(class_id, requires_grad=False) # [bs, num_classes]

        smooth_label[class_id != label_expand] = (1.0 / num_classes) * epsilon
        smooth_label[class_id == label_expand] = 1.0 - ((num_classes - 1) / num_classes * epsilon)
        smooth_label = Variable(smooth_label, requires_grad=False)  # shape: (batch_size, C)

        # get the probability
        loss = - F.log_softmax(cls_score, dim=1)  # softmax the input

        loss = torch.matmul(loss, smooth_label.permute(1, 0))  # shape (batch_size, batch_size)
        loss = torch.diag(loss, diagonal=0)  # in fact, the elements in the diagonal is the loss
        loss = torch.sum(loss)

        # average the loss
        loss = loss / float(batch_size)

    return loss


def _IoG(box_a, box_b):

    """Compute the IoG of two sets of boxes.
    Please refer the following paper for more information: https://arxiv.org/abs/1711.07752
    E.g.:
        A ∩ B / B = A ∩ B / area(B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [bs, num_a,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [bs, num_b,4]

    we first repeat the tensor to the same size, i.e.:
        [num_a, 4] -> [num_a, num_b, 4]
        [num_b, 4] -> [num_a, num_b, 4]

    Return:
        IoG: (tensor) Shape: [bs, num_a, num_b]
    """
    num_a = box_a.size()[0]
    num_b = box_b.size()[0]

    box_a_tmp = box_a.unsqueeze(dim=1)  # [num_a, 1, 4]
    box_a_tmp = box_a_tmp.repeat(1, num_b, 1)  # [num_a, num_b, 4]

    box_b_tmp = box_b.unsqueeze(dim=0)  # [1, num_b, 4]
    box_b_tmp = box_b_tmp.repeat(num_a, 1, 1)  # [num_a, num_b, 4]

    wh_b = box_b_tmp[:, :, 2:4] - box_b_tmp[:, :, 0:2]
    wh_b = torch.clamp(wh_b, min=0)
    B = torch.prod(wh_b, dim=2)  # [num_a, num_b]

    tl = torch.stack((box_a_tmp[:, :, 0:2], box_b_tmp[:, :, 0:2]), dim=0)  # [2, num_a, num_b, 2]
    tl, _ = torch.max(tl, dim=0)  # [num_a, num_b, 2]

    br = torch.stack((box_a_tmp[:, :, 2:4], box_b_tmp[:, :, 2:4]), dim=0)  # [2, num_a, num_b, 2]
    br, _ = torch.min(br, dim=0)  # [num_a, num_b, 2]

    wh = br - tl  # [ num_a, num_b, 2]
    wh = torch.clamp(wh, min=0)

    I = torch.prod(wh, dim=2)  # [ num_a, num_b]

    iog = I / B

    if iog.data.max() > 1:
        raise ValueError('Found iog > 1')

    return iog


def _IoU(box_a, box_b):
    """Compute the IoG of two sets of boxes.
    Please refer the following paper for more information: https://arxiv.org/abs/1711.07752
    E.g.:
        A ∩ B / A u B = A ∩ B / area(B u B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [bs, num_a,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [bs, num_b,4]

    we first repeat the tensor to the same size, i.e.:
        [num_a, 4] -> [num_a, num_b, 4]
        [num_b, 4] -> [num_a, num_b, 4]

    Return:
        iou: (tensor) Shape: [num_a, num_b]
    """

    num_a = box_a.size()[0]
    num_b = box_b.size()[0]

    box_a_tmp = box_a.unsqueeze(dim=1)  # [num_a, 1, 4]
    box_a_tmp = box_a_tmp.repeat(1, num_b, 1)  # [num_a, num_b, 4]

    wh_a = box_a_tmp[:, :, 2:4] - box_a_tmp[:, :, 0:2]
    wh_a = torch.clamp(wh_a, min=0)
    A = torch.prod(wh_a, dim=2)  # [num_a, num_b]

    box_b_tmp = box_b.unsqueeze(dim=0)  # [1, num_b, 4]
    box_b_tmp = box_b_tmp.repeat(num_a, 1, 1)  # [num_a, num_b, 4]

    wh_b = box_b_tmp[:, :, 2:4] - box_b_tmp[:, :, 0:2]
    wh_b = torch.clamp(wh_b, min=0)
    B = torch.prod(wh_b, dim=2)  # [num_a, num_b]

    tl = torch.stack((box_a_tmp[:, :, 0:2], box_b_tmp[:, :, 0:2]), dim=0)  # [2, num_a, num_b, 2]
    tl, _ = torch.max(tl, dim=0)  # [num_a, num_b, 2]

    br = torch.stack((box_a_tmp[:, :, 2:4], box_b_tmp[:, :, 2:4]), dim=0)  # [2, num_a, num_b, 2]
    br, _ = torch.min(br, dim=0)  # [num_a, num_b, 2]

    wh = br - tl  # [ num_a, num_b, 2]
    wh = torch.clamp(wh, min=0)

    I = torch.prod(wh, dim=2)  # [ num_a, num_b]

    iou = I / (A + B)

    if iou.data.max() > 1:
        raise ValueError('Found iou > 1')

    return iou


def _repulsion_loss(pred_box, gt_box, pred_label, pred_box_gt_index, num_gt_box, sigma_gt=1, sigma_box=0):
    """This function compute repulsion loss. This function only support the
    classification task with 2 categories
    :arg
        rois: [bs, num_rois, 4], each roi is denoted as [x1, y1, x2, y2]
        gt_box: [bs, num_gts, 4], each gt_box is denoted as [x1, y1, x2, y2]
        rois_label: [bs, num_rois], the label of rois 1 (fg) or 0 (bg)
        rois_gt_index: [bs, num_rois], the assigned gt box index for each roi

    """
    # TODO: support multi-categories classification

    # ------------------ rep gt loss ----------------------------
    bs = pred_box.size()[0]
    pred_box_gt_index = pred_box_gt_index.data.long()
    pred_label = pred_label.data.long()
    num_gt_box = num_gt_box.data.long()

    rep_gt_loss = Variable(pred_box.data.new([0]), requires_grad=True)
    count = 0
    for b in range(bs):
        gt_index = pred_box_gt_index.new(range(num_gt_box[b]))

        true_pred_box_index = ((pred_label[b] > 0) & (pred_box_gt_index[b] < num_gt_box[b]))
        true_pred_box_index = torch.nonzero(true_pred_box_index).squeeze()
        for i in true_pred_box_index:
            index = gt_index != pred_box_gt_index[b][i]
            index = torch.nonzero(index).squeeze()
            if index.size():
                one_pred_box = pred_box[b][i].unsqueeze(dim=0)
                one_gt_boxes = gt_box[b][index]

                iog = _IoG(one_pred_box, one_gt_boxes)
                iog = iog.view(-1)
                # filter out 1. Because some proposal maybe the gt box, so remove it
                index = (iog < 1).data.long()
                index = torch.nonzero(index).squeeze()

                if index.size():
                    iog = iog[index]

                    # the repulsion loss in https://github.com/bailvwangzi/repulsion_loss_ssd
                    rep_gt_loss = rep_gt_loss + iog.max()
                    count += 1

                    # the repulsion loss in origin paperhttps: https://arxiv.org/abs/1711.07752
                    # iog = iog.max()
                    # if iog.data[0] > 0:
                    #     count = count + 1
                    #     if iog.data[0] <= sigma_gt:
                    #         loss_tmp = -torch.log(1 -iog)
                    #     if iog.data[0] > sigma_gt:
                    #         loss_tmp = (iog - sigma_gt) / (1 - sigma_gt) - Variable(torch.log(iog.data.new([1-sigma_gt])))
                    #     rep_gt_loss = rep_gt_loss + loss_tmp

    if count > 0:
        rep_gt_loss = rep_gt_loss / count

    # ------------------- rep box loss -------------------------------
    rep_box_loss = Variable(pred_box.data.new([0]), requires_grad=True)
    count = 0
    for b in range(bs):
        # get the index of gt box
        pred_box_gt_index_tmp = pred_box_gt_index[b]
        if pred_box_gt_index_tmp.is_cuda:
            assigned_gt_index = np.unique(pred_box_gt_index_tmp.cpu().numpy())
        else:
            assigned_gt_index = np.unique(pred_box_gt_index_tmp.numpy())
        assigned_gt_index = pred_box_gt_index.new(assigned_gt_index)

        # the gt_index for bg rois are -1, so we need to filter out it
        assigned_gt_index = assigned_gt_index[assigned_gt_index >= 0]
        assigned_gt_index = assigned_gt_index[assigned_gt_index < num_gt_box[b]]

        # used to store the index of chosen rois for each gt box
        chosen_rois_index = []

        for gt_i in assigned_gt_index:
            index = pred_box_gt_index[b] == gt_i # each gt box has at least 1 roi
            index = torch.nonzero(index).squeeze()

            # choose one roi random
            roi_index_random = index[np.random.choice(range(index.size()[0]))]
            chosen_rois_index.append(roi_index_random)

        chosen_rois_index = pred_box_gt_index.new(chosen_rois_index)
        if chosen_rois_index.size():
            chosen_rois = pred_box[b][chosen_rois_index]

            iou = _IoU(chosen_rois, chosen_rois)
            iou = iou - torch.diag(iou.diag())

            iou = iou.view(-1)
            index = (iou > 0).data.long()
            index = torch.nonzero(index).squeeze()
            if index.size():
                iou = iou[index]
                count = count + index.size()[0]

                # we use the iou directly, without smooth
                rep_box_loss = rep_box_loss + iog.sum()

                # # the repulsion loss in origin paperhttps: https://arxiv.org/abs/1711.07752
                # index = (iou <= sigma_box).data.long()
                # index = torch.nonzero(index).squeeze()
                # if index.size():
                #     iou_1 = iou[index]
                #     iou_1 = - torch.log(1 - iou_1)
                #     loss_tmp_1 = iou_1.sum()
                # else:
                #     loss_tmp_1 = 0
                #
                # index = (iou > sigma_box).data.long()
                # index = torch.nonzero(index).squeeze()
                # if index.size():
                #     iou_2 = iou[index]
                #     iou_2 = (iou_2 - sigma_box) / (1 - sigma_box) - Variable(torch.log(iou.data.new([1-sigma_box])))
                #     loss_tmp_2 = iou_2.sum()
                # else:
                #     loss_tmp_2 = 0
                #
                # rep_box_loss = rep_box_loss + loss_tmp_1 + loss_tmp_2

    if count > 0:
        rep_box_loss = rep_box_loss / count

    return rep_gt_loss, rep_box_loss


def _deltas_and_proposals_to_bboxes(deltas, proposals, im_info):
    """
    This function obtain the prediction boxes based on the predicted deltas and proposals.
    :param deltas: Variable or tensor, [bs, num_box, 4*num_class] or [bs, num_box, 4]
        based on whether use class agnostic.
    :param proposals: [bs, num_box, 4], each proposal is denoted as [x1, y1, x2, y2]
    :return: [bs, num_box, 4]
    """
    if isinstance(deltas, Variable):
        if deltas.data.is_cuda:
            box_normalize_std = Variable(deltas.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda())
            box_normalize_mean = Variable(deltas.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda())
        else:
            box_normalize_std = Variable(deltas.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
            box_normalize_mean = Variable(deltas.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
    else:
        if deltas.is_cuda:
            box_normalize_std = deltas.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
            box_normalize_mean = deltas.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        else:
            box_normalize_std = deltas.new(cfg.TRAIN.BBOX_NORMALIZE_STDS)
            box_normalize_mean = deltas.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

    bs, num_box = deltas.size()[0], deltas.size()[1]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            deltas = deltas.view(-1, 4) * box_normalize_std + box_normalize_mean
            deltas = deltas.view(bs, proposals.size()[1], -1)
        pred_boxes = bbox_transform_inv_one_class(proposals, deltas)  # x1, y1 ,x2 ,y2

    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(proposals, (1, num_box))

    pred_boxes = clip_boxes(pred_boxes, im_info, bs)

    return pred_boxes


def _crop_pool_layer(bottom, rois, max_pool=True):
    # code modified from 
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1)
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = rois.size(0) / batch_size
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
      pre_pool_size = cfg.POOLING_SIZE * 2
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)
      crops = F.max_pool2d(crops, 2, 2)
    else:
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)
    
    return crops, grid

def _affine_grid_gen(rois, input_size, grid_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid

def _affine_theta(rois, input_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())

    # theta = torch.cat([\
    #   (x2 - x1) / (width - 1),
    #   zero,
    #   (x1 + x2 - width + 1) / (width - 1),
    #   zero,
    #   (y2 - y1) / (height - 1),
    #   (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    theta = torch.cat([\
      (y2 - y1) / (height - 1),
      zero,
      (y1 + y2 - height + 1) / (height - 1),
      zero,
      (x2 - x1) / (width - 1),
      (x1 + x2 - width + 1) / (width - 1)], 1).view(-1, 2, 3)

    return theta

def compare_grid_sample():
    # do gradcheck
    N = random.randint(1, 8)
    C = 2 # random.randint(1, 8)
    H = 5 # random.randint(1, 8)
    W = 4 # random.randint(1, 8)
    input = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
    input_p = input.clone().data.contiguous()
   
    grid = Variable(torch.randn(N, H, W, 2).cuda(), requires_grad=True)
    grid_clone = grid.clone().contiguous()

    out_offcial = F.grid_sample(input, grid)    
    grad_outputs = Variable(torch.rand(out_offcial.size()).cuda())
    grad_outputs_clone = grad_outputs.clone().contiguous()
    grad_inputs = torch.autograd.grad(out_offcial, (input, grid), grad_outputs.contiguous())
    grad_input_off = grad_inputs[0]


    crf = RoICropFunction()
    grid_yx = torch.stack([grid_clone.data[:,:,:,1], grid_clone.data[:,:,:,0]], 3).contiguous().cuda()
    out_stn = crf.forward(input_p, grid_yx)
    grad_inputs = crf.backward(grad_outputs_clone.data)
    grad_input_stn = grad_inputs[0]
    pdb.set_trace()

    delta = (grad_input_off.data - grad_input_stn).sum()
