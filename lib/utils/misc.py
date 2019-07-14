import os
import numpy as np
import torch
from torch.autograd import Variable
from lib.model.roi_align.roi_align.roi_align import RoIAlign
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import cv2



def change_to_string(input):
    """
    This function change input to a string.

    Argus:
    :param input: the parameter need to be changed to a string, it can be a dictionary,
            list and so on.
    :return: string, which is obtained from input.
    """
    output_str = input  # if input is a string
    if isinstance(input, float) or isinstance(input, int) or isinstance(input, bool):
        output_str = str(input)
    elif isinstance(input, list):
        output_str = ''
        for idx, val in enumerate(input):
            if not isinstance(val, str):
                val = change_to_string(val)
            output_str += val
            if idx != len(input) - 1:
                output_str += ', '
    elif isinstance(input, dict):
        output_str = '{\n'
        for key in input.keys():
            val = input[key]
            if not isinstance(val, str):
                val = change_to_string(val)
            output_str += ('    ' + key + ': ' + val + '\n')
        output_str += '}\n'
    return output_str


def set_visible_devices(gpus_id):
    """
    Args:
        gpus_id (list): the list of gpu to use
    """
    gpus_id_str = change_to_string(gpus_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_id_str
    print('set GPU: ' + gpus_id_str)


def load_weights(cnn_model, weights):
    """
    This function load weights of the pretrained model
    but without the RCNN_base_p branch
    argus:
    :param cnn_model: the cnn networks need to load weights
    :param weights: the pretrained weigths
    :return: no return
    """
    own_dict = cnn_model.state_dict()
    for key, val in weights.items():
        if 'RCNN_base_p' not in key and key in own_dict.keys():
            if isinstance(val, Parameter):
                val = val.data
            own_dict[key].copy_(val)
    cnn_model.load_state_dict(own_dict, strict=True)


def resize_im(im_data, im_scale, tool='roialign'):
    """
    This function resize a image. Infact the image here can not only
    images, but also can be other data. But to make sure it can work
    correctly, the image, motion vector and residual are expected.
    :param im_data: array, H x W x C, C can be 3 for image and residual
                while for motion vector, C is 2
    :param im_scale: the scale factor, the H and W of im_data is resized
                to H * im_scale and W * im_scale
    :return:
    """

    if tool == 'cv2':
        if im_data.shape[2] == 3: # image, residual
            im_resize = cv2.resize(im_data, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        elif im_data.shape[2] == 2: # mv
            pad = np.zeros(im_data.shape[0], im_data.shpe[1], 1) #
            im_data = np.concatenate((im_data, pad), axis=2) # [h, w, 3]
            im_resize = cv2.resize(im_data, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR())
            im_resize = im_resize[:, :, 0:2]
        return im_resize

    elif tool == 'roialign':
        def to_varabile(arr, requires_grad=False, is_cuda=True):
            tensor = torch.from_numpy(arr)
            if is_cuda:
                tensor = tensor.cuda()
            var = Variable(tensor, requires_grad=requires_grad)
            return var

        # the data you want
        is_cuda = False

        im_shape = im_data.shape
        H, W = im_shape[0], im_shape[1]

        im_data = np.transpose(im_data, (2, 0, 1))  # CxHxW
        im_data = np.ascontiguousarray(im_data, dtype=np.float32)
        im_data = im_data[np.newaxis]

        # x1, y1, x2, y2. the box need to crop from the im_data,
        # we will crop all the im_data since we are doing is resizing
        # im_data
        boxes_data = np.asarray([[0, 0, W, H]], dtype=np.float32)

        # the box index below is the index for the im_data, since
        # there is only one im_data, we set it to 0
        box_index_data = np.asarray([0], dtype=np.int32)

        image_torch = to_varabile(im_data, requires_grad=True, is_cuda=is_cuda)
        boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
        box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)

        # set transform_fpcoor to False is the crop_and_resize

        crop_h, crop_w = round(float(H) * im_scale), round(float(W) * im_scale)

        roi_align = RoIAlign(crop_width=crop_w, crop_height=crop_h, transform_fpcoor=True)
        croped = roi_align(image_torch, boxes, box_index)

        croped = croped[0]  # remove the batch dim, CxHxW
        croped = croped.data.numpy()  # change tensor to numpy
        croped = np.transpose(croped, (1, 2, 0))

        return croped


def load_pretrained_resnet_weights_to_our_modified_resnet(cnn_model, pretrained_weights):
    """
    argus:
    :param cnn_model: the cnn networks need to load weights
    :param weights: the pretrained weigths
    :return: no return
    """
    pre_dict = cnn_model.state_dict()
    for key, val in pretrained_weights.items():
        if key[0:5] == 'layer':
            key_list = key.split('.')
            tmp = int(int(key_list[1]) * 2)
            key_list[1] = str(tmp)
            tmp_key = ''
            for i in range(len(key_list)):
                tmp_key = tmp_key + key_list[i] + '.'

            key = tmp_key[:-1]

        if isinstance(val, Parameter):
            val = val.data

        pre_dict[key].copy_(val)
    cnn_model.load_state_dict(pre_dict)


def offsets_to_coordinates(offsets):
    """
    This function change motion vector or optical flow to corresponding coordinates in the image
    :param offsets: array_like with size [bs, h, w, 2], 2 channels denote x (0) and y (1), respectively.
    :return: array_like, with size [bs, h, w, 2], coordinates start from 0
    """

    is_numpy = isinstance(offsets, np.ndarray)
    is_double_tensor = isinstance(offsets, torch.DoubleTensor)
    is_variable = isinstance(offsets, Variable)

    if is_numpy:
        offsets = torch.FloatTensor(offsets)
    if is_double_tensor: # change to FloatTensor
        if offsets.is_cuda:
            offsets = offsets.cpu().numpy()
            offsets = torch.FloatTensor(offsets).cuda()
        else:
            offsets = offsets.numpy()
            offsets = torch.FloatTensor(offsets)

    # now offsets is a FloatTensor or Variable

    size = offsets.size()
    bs, h, w, c = size[0], size[1], size[2], size[3]

    x_coordinates = torch.FloatTensor(range(0, w))
    x_coordinates = x_coordinates.unsqueeze(dim=0)
    x_coordinates = x_coordinates.repeat(h, 1)
    x_coordinates = x_coordinates.unsqueeze(dim=0) # bs dim
    x_coordinates = x_coordinates.repeat(size[0], 1, 1)

    y_coordinates = torch.FloatTensor(range(0, h))
    y_coordinates = y_coordinates.unsqueeze(dim=1)
    y_coordinates = y_coordinates.repeat(1, w)
    y_coordinates = y_coordinates.unsqueeze(dim=0) # bs_dim
    y_coordinates = y_coordinates.repeat(size[0], 1, 1)

    coordinates_mesh = torch.FloatTensor(offsets.size())
    coordinates_mesh[:,:,:,0] = x_coordinates
    coordinates_mesh[:,:,:,1] = y_coordinates


    if offsets.is_cuda:
        coordinates_mesh = coordinates_mesh.cuda()
    if is_variable:
        coordinates_mesh = Variable(coordinates_mesh, requires_grad=False)

    coordinates = coordinates_mesh - offsets

    if is_variable: # this means the input is a Variable, offsets is also a Variable, do nothing
        pass
    if is_double_tensor: # this means the input is tensor, offsets is also a FloatTensor, do nothing
        pass
    if is_numpy: # this means the input is numpy, offsets is FloatTensor, and we convert offset to numpy
        coordinates = coordinates.numpy()

    return coordinates


def coordinates_to_flow_field(coordinates):
    """
    This function shift the coordinates to a flow field
    :param coordinates: array_like, with size [bs, h, w, 2], coordinates starts from 0
    :return: array_like, [bs, h, w, 2] that has the same size with coordinates, contain the flow
            filed. Values: x: -1, y: -1 is the left-top pixel of the input,
            and values: x: 1, y: 1 is the right-bottom pixel of the input.
    """
    is_numpy = isinstance(coordinates, np.ndarray)

    if is_numpy:
        coordinates = torch.FloatTensor(coordinates)
    # now coordinates is FloatTensor or Variable

    h, w = coordinates.size()[1], coordinates.size()[2]
    half_h, half_w = h / 2., w / 2. # float

    coordinates[:, :, :, 0] = (coordinates[:, :, :, 0] - half_w) / half_w # x
    coordinates[:, :, :, 1] = (coordinates[:, :, :, 1] - half_h) / half_h # y

    if is_numpy: # this means the inputs is a numpy, and the result is a tensor, convert it
        coordinates = coordinates.numpy()

    return coordinates


def warp_from_offsets(offsets, in_data):
    """
    This function warp the input data based on the offset. The input data must have the same
    data type.
    :param offsets: array_like, 4D, [bs x 2 x H x W], which can be motion vector or optical flow
    :param in_data: array_like, 4D, [bs x C x H x W].
    :return:
    """
    is_numpy = isinstance(offsets, np.ndarray)
    is_double_tensor = isinstance(offsets, torch.DoubleTensor)
    is_variable = isinstance(offsets, Variable)

    if is_numpy:
        offsets = torch.FloatTensor(offsets)
        in_data = torch.FloatTensor(in_data)
    if is_double_tensor:  # change to FloatTensor
        if offsets.is_cuda:
            offsets = offsets.cpu().numpy()
            offsets = torch.FloatTensor(offsets).cuda()

            in_data = in_data.cpu().numpy()
            in_data = torch.FloatTensor(in_data).cuda()
        else:
            offsets = offsets.numpy()
            offsets = torch.FloatTensor(offsets)

            in_data = in_data.numpy()
            in_data = torch.FloatTensor(in_data)

    offsets = offsets.permute(0, 2, 3, 1) # [bs x 2 x H x W] to [bs x H x W x 2]

    coord = offsets_to_coordinates(offsets)
    field = coordinates_to_flow_field(coord) # [bs x H x W x 2]

    warped = F.grid_sample(in_data, field) # the output of grid_sample is Variable

    if not is_variable:
        warped = warped.data # change to tensor

    if is_double_tensor:
        if warped.is_cuda:
            warped = offsets.cpu().numpy()
            warped = torch.DoubleTensor(warped).cuda()
        else:
            warped = warped.numpy()
            warped = torch.DoubleTensor(warped)

    if is_numpy:
        warped = warped.numpy()

    return warped






