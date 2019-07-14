import torch
from torch.autograd import Variable
from lib.model.roi_align.roi_align.roi_align import RoIAlign


def tlbr2tlwh(tlbr):
    """
    This function change the box from [x1, y1, x2, y2] to [x1, y1, w, h]
    :param tlbr: 1D tensor, [x1, y1, x2, y2]
    :return:
    """
    tlwh = tlbr.clone()
    tlwh[2:4] = tlwh[2:4] - tlwh[0:2] + 1
    return tlwh


def tlwh2tlbr(tlwh):
    """
    This function change the box from [x1, y1, w, h] to [x1, y1, x2, y2]
    :param tlbr: 1D tensor, [x1, y1, x2, y2]
    :return:
    """
    tlbr = tlwh.clone()
    tlbr[2:4] = tlbr[2:4] + tlbr[0:2] - 1
    return tlbr


def tlwh2xyah(tlwh):
    """
    This function change the box from [x1, y1, w, h] to [cx, cy, aspect_ratio, h]
    :param tlwh: 1D tensor, the box with format [x1, y1, w, h]
    :return:
    """
    xyah = tlwh.clone() # [x1, y1 ,w, h]
    xyah[0:2] = xyah[0:2] + xyah[2:4] / 2.0 - 1 # [cx, cy, w, h]
    xyah[2] /= xyah[3] # [cx, cy, w/h, h]
    return xyah


def xyah2tlwh(xyah):
    """
    This function change the bx from [cx, cy, aspect_ratio, h] to [x1, y1, w, h]
    :param xyah: 1D tensor, the box with format [cx, cy, aspect_ratio, h]
    :return:
    """
    tlwh = xyah.clone()
    tlwh[2] *= tlwh[3] # [cx, cy, w, h]
    tlwh[0:2] = tlwh[0:2] - tlwh[2:4] / 2.0 + 1 # [x1, y1, w, h]
    return tlwh


def tlbr2xyah(tlbr):
    """
    This function change the box from [x1, y1, x2, y2] to [cx, cy, aspect_ratio, h]
    :param tlbr: 1D tensor, the box with format [x1, y1, x2, y2]
    :return:
    """
    tlwh = tlbr2tlwh(tlbr)
    xyah = tlwh2xyah(tlwh)
    return xyah


def xyah2tlbr(xyah):
    """
    This function change the bx from [cx, cy, aspect_ratio, h] to [x1, y1, x2, y2]
    :param xyah: 1D tensor, the box with format [cx, cy, aspect_ratio, h]
    :return:
    """
    tlwh = xyah2tlwh(xyah)
    tlbr = tlwh2tlbr(tlwh)
    return tlbr


# def crop_data_for_boxes(boxes, in_data, crop_size=None, scale=None):
#     """
#     This function crop the corresponding data for each box using
#     ROIAlign. Single batch only!
#     :param boxes: 2D tensor or Variable with size [N, 4], [x1, y1, x2, y2].
#     :param in_data: 4D tensor or Variable, [1, :, h, w]. 1 is the number of batch,
#                 single batch only.
#     :param crop_size: array_like, size [2]. The croped size [h, w].
#     :param scale: 2D tensor or Variable, [1, 4], [f_w_scale, f_h_scale, f_w_scale, f_h_scale],
#                 used to map boxes to the in_data. If None, we will directly crop the data using
#                 the boxes.
#     :return croped_data
#     """
#     if not isinstance(in_data, Variable):
#         in_data = Variable(in_data)
#
#     if len(boxes.size()) == 1:
#         boxes = boxes.unsqueeze(dim=0)
#     if isinstance(boxes, Variable):
#         is_cuda = boxes.data.is_cuda
#     else:
#         is_cuda = boxes.is_cuda
#         boxes = Variable(boxes)
#
#     if scale is None:
#         scale = 1
#     else:
#         if not isinstance(scale, Variable):
#             scale = Variable(scale)
#         if len(scale.size()) == 1: # if f_scale has size [4]
#             scale = scale.unsqueeze(dim=0) # change to [1, 4]
#
#     # the box index below is the index for the data, since
#     # the  in_data is single batch, we set it to 0
#     box_index = torch.zeros(boxes.size()[0]).int()
#     if is_cuda:
#         box_index = box_index.cuda()
#     box_index = Variable(box_index)
#
#     boxes = boxes * scale
#
#     roi_align = RoIAlign(crop_width=crop_size[1], crop_height=crop_size[0], transform_fpcoor=True)
#     croped_data = roi_align(in_data, boxes, box_index)
#     croped_data = croped_data.data.contiguous()  # [num_box, :, h, w]
#
#     return croped_data


def crop_data_for_boxes(boxes, box_index, in_data, crop_size=None, scale=None):
    """
    This function crop the corresponding data for each box using
    ROIAlign. Single batch only!
    :param boxes: 2D tensor or Variable with size [N, 4], [x1, y1, x2, y2].
    :param in_data: 4D tensor or Variable, [1, :, h, w]. 1 is the number of batch,
                single batch only.
    :param crop_size: array_like, size [2]. The croped size [h, w].
    :param scale: 2D tensor or Variable, [1, 4], [f_w_scale, f_h_scale, f_w_scale, f_h_scale],
                used to map boxes to the in_data. If None, we will directly crop the data using
                the boxes.
    :return croped_data
    """
    if not isinstance(in_data, Variable):
        in_data = Variable(in_data)

    if len(boxes.size()) == 1:
        boxes = boxes.unsqueeze(dim=0)
    if not isinstance(boxes, Variable):
        boxes = Variable(boxes)

    if not isinstance(box_index, Variable):
        box_index = Variable(box_index)

    if scale is None:
        scale = 1
    else:
        if not isinstance(scale, Variable):
            scale = Variable(scale)
        if len(scale.size()) == 1: # if f_scale has size [4]
            scale = scale.unsqueeze(dim=0) # change to [1, 4]

    boxes = boxes * scale

    roi_align = RoIAlign(crop_width=crop_size[1], crop_height=crop_size[0], transform_fpcoor=True)
    croped_data = roi_align(in_data, boxes, box_index)
    croped_data = croped_data.data.contiguous()  # [num_box, :, h, w]

    return croped_data











