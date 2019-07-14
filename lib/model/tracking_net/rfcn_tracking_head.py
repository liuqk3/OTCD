
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.psroi_pooling.modules.psroi_pool_test import PSRoIPool
from lib.model.rpn.proposal_target_layer_cascade_gt_index import _ProposalTargetLayer
from lib.model.rpn.rpn import _RPN
from lib.model.utils.config import cfg
from lib.model.utils.net_utils import _smooth_l1_loss
from torch.autograd import Variable
import pdb

"""
The head for tracking net. We use the regression as the 
"""


class RFCN_tracking_head(nn.Module):
    """ R-FCN """
    def __init__(self, pooling_size):
        super(RFCN_tracking_head, self).__init__()

        self.pooling_size = pooling_size
        # define regression network
        self.RCNN_psroi_pool_loc = PSRoIPool(self.pooling_size, self.pooling_size, spatial_scale=1/16.0,
                                             group_size=cfg.POOLING_SIZE, output_dim=4)
        self.pooling = nn.AvgPool2d(kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE)


if __name__ == '__main__':
    print('runing rfcn_head...')
