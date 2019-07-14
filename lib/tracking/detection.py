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

    def show(self, type='im', track_id=None, show_size=(120, 40)):
        """
        This function show the history of this track.
        :param type: the type of history to show. If type is 'mask', var track_id is
                    used. Then the mask obtained between this detection and track is shown.
        :return:
        """

        if type in ['im', 'mv', 'residual']:
            feature = self.feature  # [c, h, w]
            if isinstance(feature, Variable):
                feature = feature.data

            feature = torch.norm(feature, dim=0, p=2) # [h, w]
            feature_max, _ = torch.max(feature, dim=1, keepdim=True) # [h, 1]
            feature_max, _ = torch.max(feature_max, dim=0, keepdim=True) # [1, 1]
            feature = feature / feature_max * 255
            
            if type == 'im':
                if self.im is None:
                    raise RuntimeError('Track do not have image patches to show!')
                frame_type = 0
                show_data = np.asarray(self.im.permute(1, 2, 0), dtype=np.uint8) # [h, w, c]
            elif type == 'mv':
                if self.mv is None:
                    raise RuntimeError('Track do not have motion vector patches to show!')
                frame_type = 1
                show_data = np.asarray(self.mv.permute(1, 2, 0), dtype=np.uint8) # [h, w, c]
            elif type == 'residual':
                if self.residual is None:
                    raise RuntimeError('Track do not have residual patches show!')
                frame_type = 2
                show_data = np.asarray(self.mv.permute(1, 2, 0), dtype=np.uint8)  # [h, w, c]

            show_w = show_data.shape[1]
            show_h = show_data.shape[0]
            plt.figure()
            show_data = compressed_frame_to_show(frame=show_data, frame_type=frame_type) # RGB images
            plt.subplot(2, 1, 1)
            plt.imshow(show_data)
            plt.axis('off')
            
            plt.subplot(2, 1, 2)
            feature = np.asarray(feature, dtype=np.uint8)
            feature = cv2.resize(feature, (show_w, show_h))
            plt.imshow(feature)
            plt.axis('off')
            plt.show()
        elif type in ['mask', 'masks']:

            show_h = show_size[0]
            show_w = show_size[1]

            one_mask = self.mask[str(track_id)]
            mask_t = one_mask['track'] # [num_f, c, h, w]
            mask_d = one_mask['detection']  # [num_f, c, h, w]
            show_feature_map(mask_t, save=False, show=True, show_size=(show_w, show_h))
            show_feature_map(mask_d, save=False, show=True, show_size=(show_w, show_h))

            # mask_t = torch.norm(mask_t, dim=1, p=2, keepdim=True) # [num_f, 1, h, w]
            # mask_t_max, _ = torch.max(mask_t, dim=3, keepdim=True) # [num_f, 1, h, 1]
            # mask_t_max, _ = torch.max(mask_t_max, dim=2, keepdim=True) # [num_f, 1, 1, 1]
            # mask_t = mask_t / mask_t_max
            # mask_t = mask_t * 255  # [num_f, 1, h, w]
            #
            #
            # mask_d = torch.norm(mask_d, dim=1, p=2, keepdim=True) # [num_f, 1, h, w]
            # mask_d_max, _ = torch.max(mask_d, dim=3, keepdim=True) # [num_f, 1, h, 1]
            # mask_d_max, _ = torch.max(mask_d_max, dim=2, keepdim=True) # [num_f, 1, 1, 1]
            # mask_d = mask_d / mask_d_max
            # mask_d = mask_d * 255
            #
            #
            # if isinstance(mask_t, Variable):
            #     mask_t = mask_t.data
            # if isinstance(mask_d, Variable):
            #     mask_d = mask_d.data
            #
            # num_mask = mask_t.size(0)
            # if num_mask == 0:
            #     print('Nothing to show, the number of masks is 0')
            # else:
            #     plt.figure()
            #     n_rows = 2
            #     n_cols = num_mask
            #     for i in range(num_mask):
            #         one_mask_t = mask_t[i]
            #         one_mask_t = one_mask_t.squeeze()
            #         one_mask_t = np.asarray(one_mask_t, dtype=np.uint8)
            #         one_mask_t = cv2.resize(one_mask_t, (show_w, show_h))
            #         plt.subplot(n_rows, n_cols, i+1)
            #         plt.imshow(one_mask_t)
            #         plt.axis('off')
            #
            #         one_mask_d = mask_d[i]
            #         one_mask_d = one_mask_d.squeeze()
            #         one_mask_d = np.asarray(one_mask_d, dtype=np.uint8)
            #         one_mask_d = cv2.resize(one_mask_d, (show_w, show_h))
            #         plt.subplot(n_rows, n_cols, n_cols + i + 1)
            #         plt.imshow(one_mask_d)
            #         plt.axis('off')
            #     plt.show()

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
