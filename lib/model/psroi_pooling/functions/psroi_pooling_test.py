import torch
from torch.autograd import Function
from .._ext import psroi_pooling 


class PSRoIPoolingFunction(Function):
    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim, output, mappingchannel):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)
        self.output = output
        self.mappingchannel = mappingchannel
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):

        psroi_pooling.psroi_pooling_forward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale, self.group_size, self.output_dim, \
        features, rois, self.output, self.mappingchannel)
        self.rois = rois
        self.feature_size = features.size()

        return self.output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = torch.zeros(batch_size, num_channels, data_height, data_width).cuda()

        psroi_pooling.psroi_pooling_backward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale, self.output_dim,  \
        grad_output, self.rois, grad_input, self.mappingchannel)
        return grad_input, None
