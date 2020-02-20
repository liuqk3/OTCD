
import torch
import numpy as np
from lib.tracking.utils import tlbr2tlwh
from lib.utils.visualization import compressed_frame_to_show
import matplotlib.pyplot as plt
from torch.autograd import Variable
import cv2
import math


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3

    tentative2confirmed = {'PRIVATE': 3, 'POI': 4, 'DPM': 7, 'FRCNN': 4, 'SDP': 4}
    confirmed2tentative = {'PRIVATE': 2, 'POI': 2, 'DPM': 4, 'FRCNN': 3, 'SDP': 3}
    tentative2deleted = {'PRIVATE': 10, 'POI': 10, 'DPM': 7, 'FRCNN': 10, 'SDP': 8}

    confidence_thr = {'PRIVATE': 0.99, 'POI': 0.45, 'DPM': 0.25, 'FRCNN': 0.95, 'SDP': 0.99}


class Track:
    def __init__(self, detection, track_id=None, detector_name=None):
        """
        :param xyah: bounding box, [cx, cy, w/h, h]
        :param track_id: integer, the ID of this target
        :param detector_name: the name of detector
        """
        if track_id is None:
            raise ValueError('Not assigned an ID for this track!')

        self.track_id = track_id
        self.detector_name = detector_name
        self.current_tlbr = detection.to_tlbr() # 1D tensor, [4]
        self.confidence = detection.confidence

        self.feature = detection.feature # [c, h_f, w_f]
        self.feature = self.feature.unsqueeze(dim=0) # [num_f, c, h_f, w_f], now num_f is 1
        self.mv = detection.mv # [2, h, w] or None
        if self.mv is not None:
            self.mv = self.mv.unsqueeze(dim=0) # [num_mv, 2, h, w]
        self.im = detection.im # [3, h, w] or None
        if self.im is not None:
            self.im = self.im.unsqueeze(dim=0)  # [num_im, 3, h, w], the im patch of this target
        self.residual = detection.residual # [3, h, w] or None
        if self.residual is not None:
            self.residual = self.residual.unsqueeze(dim=0)  # [num_im, 3, h, w]

        # self.has_additional_data = True
        # if self.im is None and self.mv is None and self.residual is None:
        #     self.has_additional_data = False
        #
        # if self.has_additional_data:
        #     self.mv = self.mv.unsqueeze(dim=0) # [num_mv, 2, h, w]
        #     self.im = self.im.unsqueeze(dim=0) # [num_im, 3, h, w], the im patch of this target
        #     self.residual = self.residual.unsqueeze(dim=0)  # [num_im, 3, h, w]

        self.velocity = None

        self.history_time = 24 # the history length

        # the age of this track
        self.age = 1

        # the state of the this track
        self.state = TrackState.Tentative
        if self.confidence >= TrackState.confidence_thr[self.detector_name]:
            self.state = TrackState.Confirmed

        # total number of frames since last time update
        self.time_since_update = 0 # used for confirmed2tentative

        # total number of frames since be treated as tentative
        self.time_as_tentative = 1 # used for tentative2deleted

        # total number of frames that since last time of detection
        self.time_since_detection = 0

        # total number of frames that has been detected on these frames
        self.time_been_detected = 1 # used for tentative2confirmed

    def to_tlwh(self):
        """
        :return: box of this track in the format [x1, y1, w, h]
        """
        return tlbr2tlwh(self.current_tlbr)

    def to_tlbr(self):
        """
        :return: return current box of this track in the format [x1, y1, x2, y2]
        """
        return self.current_tlbr

    def _box_transform_inv(self, tlbr, delta, sigma):
        """
        Inverse transform function to perform the inversion of transformation (the transformation
        in fast rcnn paper).
        :param tlbr: 2D tensor, [1, 4]
        :param delta: 2D tensor with size [1, 4]
        :param sigma: used for transformation and inv-transformation
        :return:
        """
        widths = tlbr[:, 2] - tlbr[:, 0] + 1.0
        heights = tlbr[:, 3] - tlbr[:, 1] + 1.0
        ctr_x = tlbr[:, 0] + 0.5 * widths
        ctr_y = tlbr[:, 1] + 0.5 * heights

        dx = delta[:, 0]
        dy = delta[:, 1]
        dw = delta[:, 2]
        dh = delta[:, 3]

        pred_ctr_x = dx * widths / (math.sqrt(2) * sigma) + ctr_x
        pred_ctr_y = dy * heights / (math.sqrt(2) * sigma) + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_x1 = pred_ctr_x - 0.5 * pred_w
        pred_y1 = pred_ctr_y - 0.5 * pred_h
        pred_x2 = pred_ctr_x + 0.5 * pred_w - 0.5
        pred_y2 = pred_ctr_y + 0.5 * pred_h - 0.5

        pred_tlbr = torch.stack((pred_x1, pred_y1, pred_x2, pred_y2), dim=1)

        return pred_tlbr

    def self_shift(self):
        """This function perform self tracking based on the motion vectors
        """
        if self.mv is not None and not self.is_deleted():
            # we just use the latest mv
            v = self.mv[-1] # dim [2, h, w], tensor
            hv = v.size(1)
            wv = v.size(2)
            hb = self.current_tlbr[2] - self.current_tlbr[0] + 1
            wb = self.current_tlbr[3] - self.current_tlbr[1] + 1

            v = v.view(2, -1) # dim [2, -1]
            v = v.mean(dim=-1) # dim [2], the shift for x and y dimension, [dx, dy]
            v[0] = v[0] / wv * wb # dx
            v[1] = v[1] / hv * hb # dy
            
            # current_tlbr: [x1, y1, x2, y2]
            self.current_tlbr[0::2] += v[0]
            self.current_tlbr[1::2] += v[1]

    def predict(self):
        """
        propagate the state of this track to current frame without the detection. This function
        is called when the track is not matched with a detection.
        :return:
        """

        if self.state == TrackState.Tentative:
            # if this track is a false detection, then we delete it
            self.time_as_tentative += 1
            if self.time_as_tentative >= TrackState.tentative2deleted[self.detector_name]:
                self.state = TrackState.Deleted
        elif self.state == TrackState.Deleted:
            # if this track is deleted, do nothing. In fact, the deleted track will not be
            # selected in Tracker
            pass
        else:
            # if this track just missed, we predict the box for current frame

            if self.velocity is not None:
                pass
                # # velocity = self.velocity[-1, :]
                # velocity = self.velocity[0, :]
                # self.current_tlbr = self._box_transform_inv(self.current_tlbr.unsqueeze(dim=0),
                #                                             velocity.unsqueeze(dim=0)).squeeze()
            else:
                # do nothing
                pass

            self.age += 1
            self.time_since_update += 1
            self.time_since_detection = 0

            if self.time_since_update >= TrackState.confirmed2tentative[self.detector_name]:
                self.state = TrackState.Tentative

    def update(self, detection, cost=None, distance_type=None):
        """
        Update the state of this track to current frame with the detection. This function is called when
        this track is matched with a detection.
        :param detection: detection.Detection
        :param cost: float, the cost that matching this track with this detection
        :param distance_type: str, which kind of distance that the cost based on.
        :return:
        """
        self.current_tlbr = detection.to_tlbr()

        self.age += 1
        self.time_since_update = 0
        self.time_since_detection = 0
        self.time_been_detected += 1

        if self.time_been_detected >= TrackState.tentative2confirmed[self.detector_name]:
            self.state = TrackState.Confirmed
            self.time_as_tentative = 0
        if detection.confidence >= TrackState.confidence_thr[self.detector_name]:
            self.time_as_tentative = 0
            self.state = TrackState.Confirmed

        # update the feature for this track
        feature = detection.feature.unsqueeze(dim=0)
        mv = detection.mv # [2, h, w] or None
        if mv is not None:
            mv = mv.unsqueeze(dim=0) # [num_mv, 2, h, w]
        im = detection.im # [3, h, w] or None
        if im is not None:
            im = im.unsqueeze(dim=0)  # [num_im, 3, h, w], the im patch of this target
        residual = detection.residual # [3, h, w] or None
        if residual is not None:
            residual = residual.unsqueeze(dim=0)  # [num_im, 3, h, w]

        #
        # im, mv, residual = None, None, None
        # if self.has_additional_data:
        #     im = detection.im.unsqueeze(dim=0)
        #     mv = detection.mv.unsqueeze(dim=0)
        #     residual = detection.residual.unsqueeze(dim=0)

        if self.feature is None:
            self.feature = feature
            self.im = im
            self.mv = mv
            self.residual = residual
        else:
            # self.feature = torch.cat((self.feature, feature), dim=0)  # [num_f, c, h_f, w_f]
            # self.feature = (1 - cost) * self.feature + cost * feature

            if distance_type == 'appearance':
                if cost > 0.2: # 0.1
                    # if the probability of the fact that this track and detection are different targets  is too large
                    pass
                else:
                    #self.feature = (1 - cost) * self.feature + cost * feature
                    self.feature = torch.cat((self.feature, feature), dim=0)
                    if self.im is not None and im is not None:
                        self.im = torch.cat((self.im, im), dim=0)
                    if self.mv is not None and mv is not None:
                        self.mv = torch.cat((self.mv, mv), dim=0)
                    if self.residual is not None and residual is not None:
                        self.residual = torch.cat((self.residual, residual), dim=0)

                    # if self.has_additional_data:
                    #     self.im = torch.cat((self.im, im), dim=0)
                    #     self.mv = torch.cat((self.mv, mv), dim=0)
                    #     self.residual = torch.cat((self.residual, residual), dim=0)
            elif distance_type == 'iou':
                # when iou is used for matching
                self.feature = torch.cat((self.feature, feature), dim=0)  # [num_f, c, h_f, w_f]
                if self.im is not None and im is not None:
                    self.im = torch.cat((self.im, im), dim=0)
                if self.mv is not None and mv is not None:
                    self.mv = torch.cat((self.mv, mv), dim=0)
                if self.residual is not None and residual is not None:
                    self.residual = torch.cat((self.residual, residual), dim=0)
                # if self.has_additional_data:
                #     self.im = torch.cat((self.im, im), dim=0)
                #     self.mv = torch.cat((self.mv, mv), dim=0)
                #     self.residual = torch.cat((self.residual, residual), dim=0)
            elif distance_type == 'joint':
                # when iou and appearance are used for matching jointly
                self.feature = torch.cat((self.feature, feature), dim=0)  # [num_f, c, h_f, w_f]
                if self.im is not None and im is not None:
                    self.im = torch.cat((self.im, im), dim=0)
                if self.mv is not None and mv is not None:
                    self.mv = torch.cat((self.mv, mv), dim=0)
                if self.residual is not None and residual is not None:
                    self.residual = torch.cat((self.residual, residual), dim=0)
                # if self.has_additional_data:
                #     self.im = torch.cat((self.im, im), dim=0)
                #     self.mv = torch.cat((self.mv, mv), dim=0)
                #     self.residual = torch.cat((self.residual, residual), dim=0)
            else:
                raise RuntimeError('Unknown distance type: {}'.format(distance_type))
        # keep at most self.history features
        history = self.feature.size(0)
        if history > self.history_time:
            self.feature = self.feature[history - self.history_time:history, :, :, :].contiguous()
            if self.mv is not None:
                self.mv = self.mv[history - self.history_time:history, :, :, :].contiguous()
            if self.im is not None:
                self.im = self.im[history - self.history_time:history, :, :, :].contiguous()
            if self.residual is not None:
                self.residual = self.residual[history - self.history_time:history, :, :, :].contiguous()

            # if self.has_additional_data:
            #     self.mv = self.mv[history - self.history_time:history, :, :, :].contiguous()
            #     self.im = self.im[history - self.history_time:history, :, :, :].contiguous()
            #     self.residual = self.residual[history - self.history_time:history, :, :, :].contiguous()

    def tracking(self, velocity=None, bbox_tlbr=None, tranform_sigma=None):
        """
        This function do tracking for this track based on the velocity. Noted that this function is called
        only when this frame is stepped over for detection.
        :param velocity: 1D tensor with size [4]
        :param tranform_sigma, used for the box transformation and inv-transformation
        :return:
        """

        if bbox_tlbr is None:
            velocity = velocity.unsqueeze(dim=0) # [1, 4]
            self.current_tlbr = self._box_transform_inv(self.current_tlbr.unsqueeze(dim=0),
                                                        velocity, tranform_sigma).squeeze()
            if self.velocity is None:
                self.velocity = velocity
            else:
                # TODO: to see whether this track is occluded
                # # compute the cosine distance between velocity and self.velocity
                # dist = torch.matmul(self.velocity, velocity.permute(1, 0)) # [n, 1]
                # norm = torch.norm(self.velocity, p=2, dim=1, keepdim=True) * torch.norm(velocity, p=2, dim=1)
                # dist_cosine = dist / norm
                #
                # if dist_cosine.max() > 0.6:
                #     self.current_tlbr = self._box_transform_inv(self.current_tlbr.unsqueeze(dim=0),
                #                                                 velocity).squeeze()
                #     self.velocity = torch.cat((self.velocity, velocity), dim=0) # [num_v, 4]
                #     num_v = self.velocity.size(0)
                #     if num_v > self.history_time:
                #         self.velocity = self.velocity[num_v - self.history_time:num_v, :]
                # else:
                #     self.current_tlbr = self._box_transform_inv(self.current_tlbr.unsqueeze(dim=0),
                #                                                 self.velocity[-1, :].unsqueeze(dim=0)).squeeze()
                self.velocity = torch.cat((self.velocity, velocity), dim=0) # [num_v, 4]
                num_v = self.velocity.size(0)
                if num_v > self.history_time:
                    self.velocity = self.velocity[num_v - self.history_time:num_v, :]
        else:
            self.current_tlbr = bbox_tlbr
        self.age += 1
        self.time_since_detection += 1

    def show_history(self, type='im'):
        """
        This function show the history of this track.
        :param type: the type of history to show
        :return:
        """
        # if not self.has_additional_data:
        if self.im is None and self.mv is None and self.residual is None:
            raise RuntimeError('Track do not have image patches or motion vectors to show!')

        feature = self.feature  # [num_f, c, h, w]
        if isinstance(feature, Variable):
            feature = feature.data

        feature = torch.norm(feature, dim=1, p=2)  # [numf, h, w]
        feature_max, _ = torch.max(feature, dim=2, keepdim=True)
        feature_max, _ = torch.max(feature_max, dim=1, keepdim=True)
        feature = feature / feature_max * 255

        if type == 'im':
            frame_type = 0
            show_data = np.asarray(self.im.permute(0, 2, 3, 1), dtype=np.uint8) # [num, h, w, c]
        elif type == 'mv':
            frame_type = 1
            show_data = np.asarray(self.mv.permute(0, 2, 3, 1), dtype=np.uint8) # [num, h, w, c]

        show_w = show_data.shape[2]
        show_h = show_data.shape[1]

        history = show_data.shape[0]
        if history == 0 :
            print('Nothing to show, the life of this track till now is {}'.format(history))
        else:
            plt.figure()
            for i in range(history):
                one_data = show_data[i]
                one_im = compressed_frame_to_show(frame=one_data, frame_type=frame_type) # RGB images
                plt.subplot(2, history, i+1)
                plt.imshow(one_im)
                plt.axis('off')

                one_f = feature[i]
                plt.subplot(2, history, history + i + 1)
                one_f = np.asarray(one_f, dtype=np.uint8)
                one_f = cv2.resize(one_f, (show_w, show_h))

                plt.imshow(one_f)
                plt.axis('off')
            plt.show()


    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted

if __name__ == "__main__":
    a = torch.FloatTensor([1,2,3,4])
    v = torch.FloatTensor([9, 10])
    a[0::2] += v
    a[1::2] += v
    print(a)


























