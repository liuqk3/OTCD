import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from lib.model.roi_align.roi_align.roi_align import RoIAlign

def prepare_data_to_show(in_data):
    # indata: [bs, c, h, w]
    if isinstance(in_data, Variable):
        in_data = in_data.data
    if in_data.is_cuda:
        in_data = in_data.cpu()

    in_data = in_data[0]
    in_data = in_data.permute(1, 2, 0)

    in_data = np.asanyarray(in_data.numpy(), dtype=np.uint8)
    return in_data


def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


# the data you want
is_cuda = False

# image_data = np.random.randn(3, 7, 5) * 100
# image_data = np.ascontiguousarray(image_data, dtype=np.float32)
# image_data = image_data[np.newaxis]
#
# boxes_data = np.asarray([[0, 0, 2, 2]], dtype=np.float32)
# box_index_data = np.asarray([0], dtype=np.int32)


frame_path = '/home/liuqk/Dataset/MOT/MOT16/train/MOT16-02/img1/000001.jpg'
image_data = plt.imread(frame_path) # HxWxC
image_data = np.transpose(image_data, (2, 0, 1)) # CxHxW
image_data = np.ascontiguousarray(image_data, dtype=np.float32)
image_data = image_data[np.newaxis]

boxes_data = np.asarray([[1, 1, 400, 600]], dtype=np.float32)
box_index_data = np.asarray([0], dtype=np.int32)


image_torch = to_varabile(image_data, requires_grad=True, is_cuda=is_cuda)
boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)

# set transform_fpcoor to False is the crop_and_resize
roi_align = RoIAlign(crop_height=400, crop_width=400, transform_fpcoor=False)
croped = roi_align(image_torch, boxes, box_index)

print(image_torch, '\n')
print(croped)

plt.figure()
plt.imshow(prepare_data_to_show(image_torch))


plt.figure()
plt.imshow(prepare_data_to_show(croped))

plt.show()






























