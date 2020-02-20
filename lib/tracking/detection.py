# vim: expandtab:ts=4:sw=4
import torch
from lib.tracking.utils import tlwh2xyah, tlwh2tlbr, tlbr2tlwh, tlbr2xyah
from lib.utils.visualization import compressed_frame_to_show, show_feature_map
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import cv2


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : 1D tensor or ndarray
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlbr : 1D tensor
        Bounding box in format `[top left x, top left y, width, height]`.
    confidence : scalar
        Detector confidence score.
    feature : tensor | NoneType
        The feature that describes the object contained in this image.
    mv: tensor | NoneType
        The motion vector for this bbox
    """

    def __init__(self, tlbr, confidence, feature=None, im=None, mv=None, residual=None):
        self.tlbr = tlbr
        self.confidence = float(confidence)
        self.feature = feature  # [c, h_f, w_f]

        self.mask = {} # used to store the masks obtained in the appearance match process

        self.mv = mv  # [2, h, w], tensor
        self.im = im  # [3, h, w], tensor, the patch of target
        self.residual = residual # [3, h, w], tensor, the patch of target

    def to_tlwh(self):
        return tlbr2tlwh(self.tlbr)

    def to_tlbr(self):
        """Convert bounding box to format `[min x, min y, max x, max y]`, i.e.,
        `[top left, bottom right]`.
        """
        # ret = self.tlwh.clone()
        # ret[2:4] += ret[0:2]
        # return ret
        return self.tlbr

    def to_xyah(self):
        """Convert bounding box to format `[center x, center y, aspect ratio,
        height]`, where the aspect ratio is `width / height`.
        """
        # ret = self.tlwh.clone()
        # ret[0:2] += ret[2:4] / 2
        # ret[2] /= ret[3]
        # return ret
        return tlbr2xyah(self.tlbr)
