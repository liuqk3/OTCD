import os
import numpy as np
import time
import torch
import coviar
import pandas
from torch.autograd import Variable

from lib.model.nms.nms_wrapper import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_one_class, bbox_iou
from lib.model.utils.blob_single import prep_im_for_blob, prep_mv_for_blob, prep_residual_for_blob
from lib.utils.misc import resize_im
from lib.utils.image_viewer import ImageViewer
from lib.utils.visualization import create_unique_color_uchar
from lib.tracking.track import Track
from lib.tracking.detection import Detection
from lib.tracking import linear_assignment
from lib.tracking import distance_metric_func
from lib.tracking.utils import crop_data_for_boxes
from lib.model.roi_align.roi_align.roi_align import RoIAlign
import matplotlib.pyplot as plt

class Tracker:
    def __init__(self, base_net_model, tracking_model, appearance_model, classes=None, args=None, cfg=None):
        self.classes = np.asarray(['__background__', 'person']) if classes is None else classes
        self.args = args
        self.cfg = cfg
        self.gop_size = 12
        self.im_viewer = None

        self.feature_crop_size = self.args.feature_crop_size  # (1024, 7, 3) the croped size of feature map, (h, w)
        self.mv_crop_size = self.args.mv_crop_size  # mv crop size. (c, h, w)
        self.im_crop_size = self.args.im_crop_size  # im crop size, (c, h, w)
        self.resdual_crop_size = self.args.residual_crop_size # residual crop size, (c, h, w)

        # the iou less than (1-self.max_iou_distance) will be disregarded.
        self.max_iou_distance = 0.7
        # the probability of been different targets larger than this value will be disregarded.
        self.max_appearance_distance = 0.25
        # the euclidean distance between the centers of two boxes (normalized by the diagonal line of this frame)
        # that larger than this threshold will be disregarded.
        self.max_euclidean_distance = 0.15 #0.2

        self.mot_dir = self.args.mot_dir # '/data0/liuqk/MOTChallenge'
        self.video_file = None  # the file path of the video need to track
        self.dataset_year = None  # MOT16, MOT17
        self.phase = None  # 'train', 'test'
        self.seq = None  # the name of sequence
        self.frame_id = None  # the frame id of current frame
        self.detector_name = None # the name of dectector
        self.tracking_thr = {'PRIVATE': 0.95, 'DPM': 0.1, 'FRCNN': 0.8, 'SDP': 0.7, 'POI': 0.33}
        self.nms_thr = {'PRIVATE': self.cfg.TEST.NMS, 'DPM': 0.3, 'FRCNN': 0.5, 'SDP': 0.6, 'POI': 0.45, }
        self._next_id = 0  # the track id
        self.tracks = []  # list, used to save tracks (track.Track)
        # list, used to save the detections (detection.Detection) need to track in current frame
        self.detections_to_track = []
        self.detections_to_save = []
        self.tracking_results = None  # used to save the tracking results
        self.detection_results = None  # used to save the detection results
        self.public_detections = None  # used to save the loaded pre_detections

        # define some variables used to do time analysis
        self.tracked_seqs = []
        self.num_frames = []  # used to store the number of frames for each video
        self.load_time = []  # used to store the time consumption of loading the image from the video
        self.detect_time = []  # used to store the time consumption of detecting
        self.associate_time = []  # used to store the time consumption of association targets with detection
        self.track_time = []  # used to store the time consumption of tracking regression
        self.offset_time = []  # used to store the time consumption of doing offsets
        self.pre_boxes_list = [] # used to store the tracked boxes in previous frames
        self.pre_boxes_list_history = 12 # the number of previous frames

        # define the mean and std for bounding-box regression
        self.bbox_reg_std = torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.bbox_reg_mean = torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        if self.args.cuda:
            self.bbox_reg_std = self.bbox_reg_std.cuda()
            self.bbox_reg_mean = self.bbox_reg_mean.cuda()

        # define the network
        self.base_net_model = base_net_model
        self.tracking_model = tracking_model
        self.appearance_model = appearance_model

        # define some tools for RoIAlign cropping
        self.im_roi_align = RoIAlign(crop_width=self.im_crop_size[2], crop_height=self.im_crop_size[1],
                                     transform_fpcoor=True)
        self.feat_roi_align = RoIAlign(crop_width=self.feature_crop_size[2], crop_height=self.feature_crop_size[1],
                                       transform_fpcoor=True)

        self.roi_align_box_index = torch.zeros(500).int()
        if self.args.cuda:
            self.roi_align_box_index = self.roi_align_box_index.cuda()
        self.roi_align_box_index = Variable(self.roi_align_box_index)

        self.cost_matrix = torch.zeros((200, 200)).float().fill_(linear_assignment.INFTY_COST).cuda()

        # define some variables used to store the input data
        self.im_data = torch.FloatTensor(1)  # used for detection or tracking
        self.im_info = torch.FloatTensor(1)  # used for detection, [h, w, im_scale, frame_type]
        self.boxes = torch.FloatTensor(1)  # used for detection or tracking
        self.num_boxes = torch.FloatTensor(1) # used for detection
        self.track_feature = torch.FloatTensor(1) # used for appearance matching
        self.detection_feature = torch.FloatTensor(1) # used for appearance matching

        if self.args.cuda:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.boxes = self.boxes.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.track_feature = self.track_feature.cuda()
            self.detection_feature = self.detection_feature.cuda()

        self.im_data = Variable(self.im_data, volatile=True)
        # the info of this frame [1, 4], [im_h, im_w, im_scale, frame_type]
        self.im_info = Variable(self.im_info, volatile=True)
        self.boxes = Variable(self.boxes, volatile=True)
        self.num_boxes = Variable(self.num_boxes, volatile=True)
        self.track_feature = Variable(self.track_feature, volatile=True)
        self.detection_feature = Variable(self.detection_feature, volatile=True)

    def prepare_data_to_show(self, in_data, tool_type='cv2'):
        # indata: [bs, c, h, w]
        if isinstance(in_data, Variable):
            in_data = in_data.data
        if in_data.is_cuda:
            in_data = in_data.cpu()

        in_data = in_data[0]
        in_data = in_data.permute(1, 2, 0) # h w c, BGR channel

        if tool_type == 'cv2':
            pass
        elif tool_type == 'plt':
            in_data = in_data[:, :, [2, 1, 0]]

        in_data = np.asanyarray(in_data.numpy(), dtype=np.uint8)
        return in_data

    def _crop_data_for_boxes(self, boxes, in_data, scale=None, in_data_type='feature'):
        """
        This function crop the corresponding data for each box using
        ROIAlign. Single batch only!
        :param boxes: 2D tensor or Variable with size [N, 4], [x1, y1, x2, y2].
        :param in_data: 4D tensor or Variable, [1, :, h, w]. 1 is the number of batch,
                    single batch only.
        :param scale: 2D tensor or Variable, [1, 4], [f_h_scale, f_w_scale, f_h_scale, f_w_scale],
                    used to map boxes to the in_data. If None, we will directly crop the data using
                    the boxes.
        :param in_data_type: str, the type of input data, 'feature', 'im', 'mv', 'residual'
        :return croped_data
        """
        if not isinstance(in_data, Variable):
            in_data = Variable(in_data)

        if not isinstance(boxes, Variable):
            boxes = Variable(boxes)

        if len(boxes.size()) == 1:
            boxes = boxes.unsqueeze(dim=0)

        if scale is None:
            scale = 1
        else:
            if not isinstance(scale, Variable):
                scale = Variable(scale)
            if len(scale.size()) == 1: # if f_scale has size [4]
                scale = scale.unsqueeze(dim=0) # change to [1, 4]

        # the box index below is the index for the data, since
        # the  in_data is single batch, we set it to 0
        box_index = self.roi_align_box_index[:boxes.size(0)]
        boxes = boxes * scale
        if in_data_type == 'feature':
            croped_data = self.feat_roi_align(in_data, boxes, box_index)
        else: # crop the image, mv, residual patch
            croped_data = self.im_roi_align(in_data, boxes, box_index)
        croped_data = croped_data.data.contiguous()  # [num_box, :, h, w]

        return croped_data

    def reset(self, tracking_output_file=None, detection_output_file=None):
        """
        This function reset the tracker, so it can track on the next video
        :param tracking_output_file: the file path to save tracking results
        :param detection_output_file: the file path to save detection results
        :return:
        """

        # before reset the tracker, we first save the tracking and detection results
        fmt = ['%d', '%d', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%d', '%d', '%d']  # the format to save the results
        if self.args.cuda:
            if self.tracking_results is None:
                self.tracking_results = torch.zeros(1, 10)
            else:
                self.tracking_results = self.tracking_results.cpu()

            if self.detection_results is None:
                self.detection_results = torch.zeros(1, 10)
            else:
                self.detection_results = self.detection_results.cpu()

        # save the detection as pth file
        if tracking_output_file is not None:
            if os.path.exists(tracking_output_file):
                os.remove(tracking_output_file)
            tracking_results = self.tracking_results.numpy()
            np.savetxt(tracking_output_file, tracking_results, fmt=fmt, delimiter=',')
            print('tracking results saved in ' + tracking_output_file)

        if detection_output_file is not None and self.detection_results is not None:
            if self.args.save_detections_with_feature:
                detection_output_file_pth = detection_output_file.split('txt')
                detection_output_file_pth = detection_output_file_pth[0] + 'pth'
                torch.save(self.detection_results.contiguous(), detection_output_file_pth)
                print('detections results saved in ' + detection_output_file)

            if os.path.exists(detection_output_file):
                os.remove(detection_output_file)

            detection_results = self.detection_results.numpy()
            np.savetxt(detection_output_file, detection_results[:, 0:10], fmt=fmt, delimiter=',')
            print('detections results saved in ' + detection_output_file)

        # reset the tracker
        self.im_viewer = None
        self.frame_id = None
        self._next_id = 0  # the track id
        self.tracks = []  # list, used to save tracks (track.Track)
        self.tracking_results = None  # used to save the tracking results
        self.detections_to_track = []  # list, used to save the detections (detection.Detection) of current frame
        self.detections_to_save = []
        self.detection_results = None  # used to save the detection results
        self.pre_boxes_list = []

    def save_time(self, time_file):
        """
        This function save the time collected untill this function is called.
        :param time_file:
        :return:
        """
        # time analysis
        if len(self.tracked_seqs) == 0:
            seqs = np.array([''])
        else:
            seqs = np.array(self.tracked_seqs)

        self.tracked_seqs = []

        if len(self.num_frames) == 0:
            num_frames = np.array([0])
        else:
            num_frames = np.array(self.num_frames)

        self.num_frames = []

        if len(self.load_time) == 0:
            load_time = np.array([0])
        else:
            load_time = np.array(self.load_time)

        self.load_time = []

        if len(self.detect_time) == 0:
            detect_time = np.array([0])
        else:
            detect_time = np.array(self.detect_time)

        self.detect_time = []

        if len(self.associate_time) == 0:
            associate_time = np.array([0])
        else:
            associate_time = np.array(self.associate_time)

        self.associate_time = []

        if len(self.track_time) == 0:
            track_time = np.array([0])
        else:
            track_time = np.array(self.track_time)

        self.track_time = []

        if len(self.offset_time) == 0:
            offset_time = np.array([0])
        else:
            offset_time = np.array(self.offset_time)

        self.offset_time = []

        total_time_load = load_time.sum() + detect_time.sum() + associate_time.sum() + track_time.sum() + offset_time.sum()
        total_frames = len(load_time)

        total_time_no_load = detect_time.sum() + associate_time.sum() + track_time.sum() + offset_time.sum()

        if os.path.exists(time_file):
            os.remove(time_file)
        f = open(time_file, 'w')

        print('sequences:\n', seqs,
              '\n\nnumber of frames:\n', num_frames,
              '\n\ntotal number frames: {}'.format(num_frames.sum()),
              '\n\naverage load time: {}/{} = {}s'.format(load_time.sum(), load_time.shape[0], load_time.mean()),
              '\n\naverage detect time: {}/{} = {}s'.format(detect_time.sum(), detect_time.shape[0],
                                                            detect_time.mean()),
              '\n\naverage associate time: {}/{} = {}s'.format(associate_time.sum(), associate_time.shape[0],
                                                               associate_time.mean()),
              '\n\naverage track time: {}/{} = {}'.format(track_time.sum(), track_time.shape[0], track_time.mean()),
              '\n\naverage offset time: {}/{} = {}'.format(offset_time.sum(), offset_time.shape[0], offset_time.mean()),
              '\n\naverage time per frame (with load) = {}/{} = {}'.format(total_time_load, total_frames,
                                                                           total_time_load / total_frames),
              '\naverage time per frame (without load) = {}/{} = {}'.format(total_time_no_load, total_frames,
                                                                            total_time_no_load / total_frames),
              '\n\nFPS (with load): {}'.format(1.0 / (total_time_load / total_frames)),
              '\nFPS (without load): {}'.format(1.0 / (total_time_no_load / total_frames)),
              file=f)

        print('time analysis file saved in ' + time_file)
        print('FPS (without load): {}'.format(1.0 / (total_time_no_load / total_frames)))

    def get_frame_blob_from_video(self, video_path, frame_id, load_data_for=None):
        """
        This function extract the image, motion vector, residual from the given video
        and the frame id.
        :param video_path: string, the path to the raw video (.mp4)
        :param frame_id: int, the frame id
        :param load_data_for: str, determine to load the data for tracking or detection
        :return: blob: 3D array, [h, w, 3+2+3]
        :return: im_scale: float(target_size) / float(im_size_min)
        """

        accumulate = False

        gop_idx = int((frame_id - 1) / self.gop_size)  # GOP starts from 0, while frame_id  here starts from 1.
        in_group_idx = int((frame_id - 1) % self.gop_size)  # the index in the group

        if load_data_for == 'track': # load mv and residual
            mv = coviar.load(video_path, gop_idx, in_group_idx, 1, accumulate)
            residual = coviar.load(video_path, gop_idx, in_group_idx, 2, accumulate)

            # check whether it is a gray image
            if len(residual.shape) == 2:
                residual = residual[:, :, np.newaxis]
                residual = np.concatenate((residual, residual, residual), axis=2)

            residual_shape = residual.shape
            residual_size_min = np.min(residual_shape[0:2])
            residual_size_max = np.max(residual_shape[0:2])

            target_size = self.cfg.TEST.SCALES[0]  # cfg.TEST.SCALES = (600, )
            # Prevent the biggest axis from being more than MAX_SIZE
            residual_scale = float(target_size) / float(residual_size_min)
            if np.round(residual_scale * residual_size_max) > self.cfg.TEST.MAX_SIZE:
                im_scale = float(self.cfg.TEST.MAX_SIZE) / float(residual_size_max)
                target_size = np.round(im_scale * residual_size_min)

            mv, mv_scale = prep_mv_for_blob(im=mv,
                                            mv_normal_scale=self.cfg.MV_NORMAL_SCALE,
                                            mv_means=self.cfg.MV_MEANS,
                                            mv_stds=self.cfg.MV_STDS,
                                            target_size=target_size,
                                            channel=self.cfg.MV_CHANNEL)
            residual, residual_scale = prep_residual_for_blob(im=residual,
                                                              pixel_normal_scale=self.cfg.RESIDUAL_NORMAL_SCALE,
                                                              pixel_means=self.cfg.RESIDUAL_MEANS,
                                                              pixel_stds=self.cfg.RESIDUAL_STDS,
                                                              target_size=target_size,
                                                              channel=self.cfg.RESIDUAL_CHANNEL)

            # check the scales of im, mv and residual
            if mv_scale != residual_scale:
                raise RuntimeError(
                    'the scales to resize motion vector {} and residual {} are not the same'.
                        format(mv_scale, residual_scale))

            residual_shape = residual.shape
            if self.args.tracking_net_data_type == 'mv_residual':
                frame_data = np.zeros((residual_shape[0], residual_shape[1], 2 + 3))
                frame_data[:, :, 0:2] = mv
                frame_data[:, :, 2:5] = residual
            elif self.args.tracking_net_data_type == 'mv':
                frame_data = mv
            elif self.args.tracking_net_data_type == 'residual':
                frame_data = residual

            return frame_data, residual_scale

        elif load_data_for in ['base_feat', 'detect']: # load im (processed)
            im = coviar.load(video_path, gop_idx, in_group_idx, 0, accumulate)

            # check whether it is a gray image
            if len(im.shape) == 2:
                im = im[:, :, np.newaxis]
                im = np.concatenate((im, im, im), axis=2)

            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size = self.cfg.TEST.SCALES[0]  # cfg.TEST.SCALES = (600, )
            # Prevent the biggest axis from being more than MAX_SIZE
            im_scale = float(target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self.cfg.TEST.MAX_SIZE:
                im_scale = float(self.cfg.TEST.MAX_SIZE) / float(im_size_max)
                target_size = np.round(im_scale * im_size_min)

            im_data, im_scale = prep_im_for_blob(im=im,
                                                 pixel_normal_scale=self.cfg.PIXEL_NORMAL_SCALE,
                                                 pixel_means=self.cfg.PIXEL_MEANS,
                                                 pixel_stds=self.cfg.PIXEL_STDS,
                                                 target_size=target_size,
                                                 channel=self.cfg.PIXEL_CHANNEL)
            return im_data, im_scale

        elif load_data_for == 'vis': # load im (no processed)
            im = coviar.load(video_path, gop_idx, in_group_idx, 0, accumulate)
            # check whether it is a gray image
            if len(im.shape) == 2:
                im = im[:, :, np.newaxis]
                im = np.concatenate((im, im, im), axis=2)

            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size = self.cfg.TEST.SCALES[0]  # cfg.TEST.SCALES = (600, )
            # Prevent the biggest axis from being more than MAX_SIZE
            im_scale = float(target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self.cfg.TEST.MAX_SIZE:
                im_scale = float(self.cfg.TEST.MAX_SIZE) / float(im_size_max)

            im = resize_im(im, im_scale)
            return im
        elif load_data_for == 'crop': # load im (no processed) and mv
            im = coviar.load(video_path, gop_idx, in_group_idx, 0, accumulate)
            mv = coviar.load(video_path, gop_idx, in_group_idx, 1, accumulate)
            residual = coviar.load(video_path, gop_idx, in_group_idx, 2, accumulate)
            # check whether it is a gray image
            if len(im.shape) == 2:
                im = im[:, :, np.newaxis]
                im = np.concatenate((im, im, im), axis=2)

                residual = residual[:, :, np.newaxis]
                residual = np.concatenate((residual, residual, residual), axis=2)

            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size = self.cfg.TEST.SCALES[0]  # cfg.TEST.SCALES = (600, )
            # Prevent the biggest axis from being more than MAX_SIZE
            im_scale = float(target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self.cfg.TEST.MAX_SIZE:
                im_scale = float(self.cfg.TEST.MAX_SIZE) / float(im_size_max)

            im = resize_im(im, im_scale)
            residual = resize_im(residual, im_scale)
            mv = mv / im_scale
            mv = resize_im(mv, im_scale)

            return im, mv, residual

    def visualize_results(self, kind_of_boxes='tracking'):
        """
        This function show tracking or detection results in real time.
        :param kind_of_boxes: str, 'tracking', 'detection' or 'both'
        :return:
        """

        # get boxes
        boxes = []
        if kind_of_boxes in ['tracking', 'tracks']:
            for t in self.tracks:
                if t.is_confirmed():
                    tlwh = t.to_tlwh()
                    one_box = [t.track_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], round(t.confidence, 3)]
                    boxes.append(one_box)
        elif kind_of_boxes in ['detection', 'detections']:
            for d in self.detections_to_track:
                    tlwh = d.to_tlwh()
                    one_box = [-1, tlwh[0], tlwh[1], tlwh[2], tlwh[3], round(d.confidence, 3)]
                    boxes.append(one_box)
        else:
            for t in self.tracks:
                if t.is_confirmed():
                    tlwh = t.to_tlwh()
                    one_box = [t.track_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], round(t.confidence, 3)]
                    boxes.append(one_box)

            for d in self.detections_to_track:
                tlwh = d.to_tlwh()
                one_box = [-1, tlwh[0], tlwh[1], tlwh[2], tlwh[3], round(d.confidence, 3)]
                boxes.append(one_box)

        # prepare image
        im_data = self.get_frame_blob_from_video(video_path=self.video_file,
                                                 frame_id=self.frame_id,
                                                 load_data_for='vis')
        im_data = np.asarray(im_data, dtype=np.uint8)

        if self.im_viewer is None:
            im_shape = im_data.shape
            h, w = im_shape[0], im_shape[1]
            self.im_viewer = ImageViewer(update_ms=1, window_shape=(w, h))

        self.im_viewer.image = im_data.copy()
        self.im_viewer.annotate(20, 30, str(self.frame_id) + '/' + str(self.num_frames[-1]), color=(159, 255, 84))

        boxes = np.asarray(boxes)
        self.im_viewer.thickness = 2
        for box in boxes:
            target_id = int(box[0])
            tlwh = box[1: 5]
            if len(box) > 5:
                confidence = str(box[5])
            else:
                confidence = None

            if target_id <= 0:  # detection
                # self.viewer.color = create_unique_color_uchar(random.randint(-100, 100))
                self.im_viewer.color = 0, 0, 255
                self.im_viewer.rectangle(x=tlwh[0], y=tlwh[1], w=tlwh[2], h=tlwh[3], label_br=confidence)
            else:  # gt or track results
                self.im_viewer.color = create_unique_color_uchar(target_id)
                self.im_viewer.rectangle(x=tlwh[0], y=tlwh[1], w=tlwh[2], h=tlwh[3], label_tl=str(target_id),
                                         label_br=confidence)

        # show image
        # self.im_viewer.run()
        self.im_viewer.show_image()

    def _get_previous_boxes(self):
        """
        This function get the boxes in previous frames
        :return: 3D variable or None
        """
        # get the boxes in previous frame
        pre_boxes = None
        for t in self.tracks:
            one_box = t.to_tlbr()
            # rescale to origin image
            # one_box = one_box / im_scale
            one_box = one_box.unsqueeze(dim=0)  # [1, 4], [x1, y1, x2, y2]
            batch_indx = one_box.new([[0]])
            one_box = torch.cat((batch_indx, one_box), dim=1)

            if pre_boxes is None:
                pre_boxes = one_box.new().resize_(0, 4)

            pre_boxes = torch.cat((pre_boxes, one_box), dim=0)  # [num_tracks, 4]
        if pre_boxes is not None:
            pre_boxes = pre_boxes.unsqueeze(dim=0)  # [bs ,num_track, 4], here bs = 1
            self.pre_boxes_list.append(pre_boxes)

        if len(self.pre_boxes_list) > 0:
            self.pre_boxes_list = self.pre_boxes_list[-self.pre_boxes_list_history:]
            pre_boxes = torch.cat(self.pre_boxes_list, dim=1)
            self.boxes.data.resize_(pre_boxes.size()).copy_(pre_boxes).contiguous()
            pre_boxes = self.boxes.clone() # shift to GPU if necessary and change to Variable

        return pre_boxes

    def _proposal_prob_delta_to_boxes(self, proposal, cls_prob, delta, after_nms):
        """
        This function get the boxes based on the proposals, predicted deltas and scores.
        Note that the proposal, delta and score are the output of head of detector
        NMS will be applied.
        :param proposal, 3D tensor, with size [bs, num_box, 5], each boxes is denoted
                as [batch_ind, x1, y1, x2, y2]
        :param cls_prob: 3D tensor, with size [bs, num_box, num_cls]. In our case, num_cls == 2
        :param delta, 3D tensor, with size [bs, num_box, 4 x num_cls] or [bs, num_box, 4].
                The predicted deltas for each proposal.
        :param after_nms: bool, if True, the outputted boxes are processed by nms
        :return: 3D tensor with size [bs, num_box, 5], each row is [x1, y1 ,x2, y2, score]
        """
        if isinstance(proposal, Variable):
            proposal = proposal.data  # [bs. N, 5]
        if isinstance(cls_prob, Variable):
            cls_prob = cls_prob.data  # [bs, N ,2]
        if isinstance(delta, Variable):
            delta = delta.data  # [bs, N, 8] for agnostic=False

        scores = cls_prob.clone()
        boxes = proposal[:, :, 1:5]

        batch_size = boxes.size(0)  # here it is 1
        if self.cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = delta.clone()
            if self.cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * self.bbox_reg_std + self.bbox_reg_mean
                if self.args.class_agnostic:
                    box_deltas = box_deltas.view(batch_size, -1, 4)
                else:
                    box_deltas = box_deltas.view(batch_size, -1, 4 * len(self.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas)  # x1, y1 ,x2 ,y2
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes = clip_boxes(pred_boxes, self.im_info.data, pred_boxes.size()[0])

        scores = scores.squeeze(dim=0)  # remove the batch index, [num_box, 2]
        pred_boxes = pred_boxes.squeeze(dim=0)  # remove the batch index, [num_box, 2*4]

        idx_person = 1  # the person class id in motchallenge
        inds = torch.nonzero(scores[:, idx_person] > 0).view(-1)  # we keep all boxes before NMS

        # if there is detections for person
        cls_dets = None
        if inds.numel() > 0:
            if self.args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, idx_person * 4:(idx_person + 1) * 4]

            cls_scores = scores[:, idx_person][inds]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

            _, order = torch.sort(cls_scores, 0, True)
            cls_dets = cls_dets[order]
            if after_nms:
                keep = nms(cls_dets, self.nms_thr[self.detector_name], force_cpu=not self.cfg.USE_GPU_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]  # x1, y1 ,x2, y2, score
            cls_dets = cls_dets.unsqueeze(dim=0)  # 3D tensor

        return cls_dets

    def do_detection_public(self):
        """
        This function do detection on one frame
        :return: the output of detection_model, and the load time, detect time
        """

        frame_type = 0 if int((self.frame_id - 1) % self.gop_size) == 0 else 1

        t1 = time.time()
        im_data_tmp, im_scale_tmp = self.get_frame_blob_from_video(video_path=self.video_file,
                                                                   frame_id=self.frame_id,
                                                                   load_data_for='base_feat')
        load_time = time.time() - t1

        im_info_tmp = np.array([[im_data_tmp.shape[0], im_data_tmp.shape[1], im_scale_tmp, frame_type]],
                               dtype=np.float32)
        im_info_tmp = torch.from_numpy(im_info_tmp)
        im_data_tmp = np.array(im_data_tmp[np.newaxis, :, :, :], dtype=np.float32)  # [bs, h, w, c]
        im_data_tmp = torch.from_numpy(im_data_tmp).permute(0, 3, 1, 2).contiguous()  # [bs, c, h, w]

        self.im_info.data.resize_(im_info_tmp.size()).copy_(im_info_tmp).contiguous()
        self.im_data.data.resize_(im_data_tmp.size()).copy_(im_data_tmp).contiguous()

        # get the detections on current frame
        cur_boxes = None
        idx = self.public_detections[:, 0] == self.frame_id
        idx = torch.nonzero(idx).squeeze().long()
        if idx.size():
            cur_boxes = self.public_detections[idx, :]
            cur_boxes = cur_boxes[:, 2:7].clone()  # [x1, y1 ,w ,h, score]
            cur_boxes[:, 2:4] = cur_boxes[:, 0:2] + cur_boxes[:, 2:4] - 1  # x1, y1 ,x2, y2


            #cur_boxes = torch.FloatTensor([[1561, 227, 1770, 915, 1]]).cuda() # frame MOT16-09, 110
            #cur_boxes = torch.FloatTensor([[1355, 197, 1719, 929, 1]]).cuda() # frame MOT16-09, 120

            #cur_boxes = torch.FloatTensor([[1249, 381, 1541, 1080, 1]]).cuda() # frame MOT16-08, 48
            #cur_boxes = torch.FloatTensor([[1048, 380, 1271, 1003, 1]]).cuda() # frame MOT16-08, 66

            cur_boxes[:, 0:4] = cur_boxes[:, 0:4] * im_scale_tmp
            cur_boxes = cur_boxes.unsqueeze(dim=0)  # the batch index

        pre_boxes = self._get_previous_boxes()

        t2 = time.time()
        if pre_boxes is None:
            output = self.base_net_model(self.im_data)
            feature_map, f_scale = output[0], output[1]
            self.detections_to_save, self.detections_to_track = self._detection_bbox_to_detection_list(det_bboxes=cur_boxes,
                                                                                                       feature_map=feature_map,
                                                                                                       f_scale=f_scale)
        else:
            output = self.base_net_model(self.im_data, pre_boxes)
            rois, cls_prob, delta, feature_map, f_scale = output[0], output[1], output[2], output[3], output[4]

            pre_boxes = self._proposal_prob_delta_to_boxes(proposal=rois, cls_prob=cls_prob, delta=delta, after_nms=True)  # 3D
            pre_boxes = pre_boxes.squeeze(dim=0)

            inds = pre_boxes[:, 4] >= self.tracking_thr['PRIVATE']
            inds = torch.nonzero(inds).squeeze().long()

            if inds.size():
                pre_boxes = pre_boxes[inds, :]
                pre_boxes[:, 4] = self.tracking_thr[self.detector_name]  # the score as low as much
            else:
                # pre_boxes = None
                pre_boxes = None

            if pre_boxes is None:
                if cur_boxes is None:
                    boxes = None
                else:
                    boxes = cur_boxes.squeeze(dim=0)
            else:
                if cur_boxes is None:
                    boxes = pre_boxes
                else:
                    boxes = torch.cat((pre_boxes, cur_boxes.squeeze(dim=0)), dim=0)
            if boxes is not None:
                _, order = torch.sort(boxes[:, 4], 0, True)
                boxes = boxes[order]

                # keep = nms(boxes, self.cfg.TEST.NMS, force_cpu=not self.cfg.USE_GPU_NMS)
                keep = nms(boxes, self.nms_thr[self.detector_name], force_cpu=not self.cfg.USE_GPU_NMS)
                boxes = boxes[keep.view(-1).long()]  # x1, y1 ,x2, y2, score
                boxes = boxes.unsqueeze(dim=0)  # 3D tensor

            self.detections_to_save, self.detections_to_track = self._detection_bbox_to_detection_list(det_bboxes=boxes,
                                                                                                       feature_map=feature_map,
                                                                                                       f_scale=f_scale)

        detect_time = time.time() - t2

        return load_time, detect_time

    def do_detection_private(self):
        """
        This function do detection on one frame
        :return: the output of detection_model, and the load time, detect time
        """

        frame_type = 0 if int((self.frame_id - 1) % self.gop_size) == 0 else 1

        t1 = time.time()
        im_data_tmp, im_scale_tmp = self.get_frame_blob_from_video(video_path=self.video_file,
                                                                   frame_id=self.frame_id,
                                                                   load_data_for='detect')
        load_time = time.time() - t1

        im_info_tmp = torch.FloatTensor([[im_data_tmp.shape[0], im_data_tmp.shape[1], im_scale_tmp, frame_type]])
        im_data_tmp = np.array(im_data_tmp[np.newaxis, :, :, :], dtype=np.float32)  # [bs, h, w, c]
        im_data_tmp = torch.from_numpy(im_data_tmp).permute(0, 3, 1, 2).contiguous()  # [bs, c, h, w]
        gt_boxes_tmp = torch.zeros(1, 1, 5)
        num_boxes_tmp = torch.zeros(1)

        self.im_info.data.resize_(im_info_tmp.size()).copy_(im_info_tmp).contiguous()
        self.im_data.data.resize_(im_data_tmp.size()).copy_(im_data_tmp).contiguous()
        self.boxes.data.resize_(gt_boxes_tmp.size()).copy_(gt_boxes_tmp).contiguous()
        self.num_boxes.data.resize_(num_boxes_tmp.size()).copy_(num_boxes_tmp).contiguous()

        # get the boxes in previous frame
        pre_boxes = self._get_previous_boxes()

        t2 = time.time()
        output = self.base_net_model(self.im_info, self.im_data, self.boxes, self.num_boxes, pre_boxes)
        rois, cls_prob, delta, feature_map, f_scale = output[0], output[1], output[2], output[3], output[4]

        det_bboxes = self._proposal_prob_delta_to_boxes(proposal=rois,
                                                        cls_prob=cls_prob,
                                                        delta=delta, after_nms=True)
        self.detections_to_save, self.detections_to_track = self._detection_bbox_to_detection_list(det_bboxes=det_bboxes,
                                                                                                   feature_map=feature_map,
                                                                                                   f_scale=f_scale)
        detect_time = time.time() - t2

        # print("detecrtion time: {}".format(detect_time))
        return load_time, detect_time

    def _detection_bbox_to_detection_list(self, det_bboxes, feature_map, f_scale):
        """
        This function obtain the box from the output of the cnn_model. Noted that
        the returned boxes are not clipped and they are the coordinates of the
        testing image. So if you want the coordinates of the origin image, you need
        to clip and resize the boxes.

        :param det_bboxes: 3D tensor with size [bs, num_box, 5]
        :param feature_map: 4D tensor, with size [bs, h, w ,c]
        :param f_scale: 1D tensor, with size [4]. The scales that
               obtained based on the size of im_data and featurer, [4]
               used to map rois from im_data to feature map
        :return: a list, each element in it is a 1D tensor, (x1, y1 ,x2 ,y2, score)
        """
        # if there is detections for person
        detections_to_save = []
        detections_to_track = []

        if det_bboxes is not None:
            det_bboxes = det_bboxes.squeeze(dim=0) # remove the batch dim
            croped_im, croped_mv, croped_res, croped_f = None, None, None, None

            if self.args.iou_or_appearance in ['both', 'appearance', 'iou']:
                # feature map [bs, channels, h, w]. Noted that h, w here is not those in mv
                croped_f = self._crop_data_for_boxes(boxes=det_bboxes[:, 0:4], in_data=feature_map,
                                                     scale=f_scale, in_data_type='feature')

            if self.args.additional_data_for_box:
                im, mv, residual = self.get_frame_blob_from_video(video_path=self.video_file,
                                                                  frame_id=self.frame_id,
                                                                  load_data_for='crop')
                im = torch.FloatTensor(im).contiguous()
                mv = torch.FloatTensor(mv).contiguous()
                residual = torch.FloatTensor(residual).contiguous()
                im = im.unsqueeze(dim=0).permute(0, 3, 1, 2)  # [bs, 3, h, w]
                mv = mv.unsqueeze(dim=0).permute(0, 3, 1, 2)  # [bs, 2, h, w]
                residual = residual.unsqueeze(dim=0).permute(0, 3, 1, 2)  # [bs, 3, h, w]

                if self.args.cuda:
                    im = im.cuda()
                    mv = mv.cuda()
                    residual = residual.cuda()
                croped_im = self._crop_data_for_boxes(boxes=det_bboxes[:, 0:4], in_data=im, in_data_type='im')

                plt.figure()
                plt.imshow(self.prepare_data_to_show(croped_im, tool_type='plt'))
                plt.show()

                croped_mv = self._crop_data_for_boxes(boxes=det_bboxes[:, 0:4], in_data=mv, in_data_type='mv')
                croped_res = self._crop_data_for_boxes(boxes=det_bboxes[:, 0:4], in_data=residual,
                                                       in_data_type='residual')

            for i in range(det_bboxes.size()[0]):
                one_bbox = det_bboxes[i]  # [x1, y1, x2, y2, score]
                one_im = None if croped_im is None else croped_im[i]  # [2, h, w]
                one_mv = None if croped_mv is None else croped_mv[i]  # [2, h, w]
                one_res = None if croped_res is None else croped_res[i]
                one_f = None if croped_f is None else croped_f[i]  # [c, h, w]
                one_detection = Detection(tlbr=one_bbox[0:4],
                                          confidence=one_bbox[4],
                                          feature=one_f,
                                          mv=one_mv,
                                          im=one_im,
                                          residual=one_res)
                detections_to_save.append(one_detection)
                if one_detection.confidence >= self.tracking_thr[self.detector_name]:
                    detections_to_track.append(one_detection)

        return detections_to_save, detections_to_track

    def do_tracking(self):
        """
        This function do tracking regression for all targets in
        self.tracks.
        :return:
        """

        # get the boxes in last frame
        if len(self.tracks) > 0:
            boxes = None
            for t in self.tracks:
                one_box = t.to_tlbr()
                # rescale to origin image
                # one_box = one_box / im_scale
                one_box = one_box.unsqueeze(dim=0)  # [1, 4], [x1, y1, x2, y2]
                if boxes is None:
                    boxes = one_box.new().resize_(0, 4)

                boxes = torch.cat((boxes, one_box), dim=0)  # [num_tracks, 4]

            boxes = boxes.unsqueeze(dim=0)  # [bs ,num_track, 4], here bs = 1
            self.boxes.data.resize_(boxes.size()).copy_(boxes).contiguous()

            # load the im_data
            frame_type = 0 if int((self.frame_id - 1) % self.gop_size) == 0 else 1
            t1 = time.time()
            im_data_tmp, im_scale_tmp = self.get_frame_blob_from_video(video_path=self.video_file,
                                                                       frame_id=self.frame_id,
                                                                       load_data_for='track')
            load_time = time.time() - t1

            im_info_tmp = np.array([[im_data_tmp.shape[0], im_data_tmp.shape[1], im_scale_tmp, frame_type]],
                                   dtype=np.float32)
            im_info_tmp = torch.from_numpy(im_info_tmp)
            im_data_tmp = np.array(im_data_tmp[np.newaxis, :, :, :], dtype=np.float32)  # [bs, h, w, c]
            im_data_tmp = torch.from_numpy(im_data_tmp).permute(0, 3, 1, 2).contiguous()  # [bs, c, h, w]

            self.im_info.data.resize_(im_info_tmp.size()).copy_(im_info_tmp).contiguous()
            self.im_data.data.resize_(im_data_tmp.size()).copy_(im_data_tmp).contiguous()

            t2 = time.time()
            output = self.tracking_model(self.boxes, self.im_data)  # the deltas, [bs, num_box, 4]
            track_time = time.time() - t2

            return output, load_time, track_time
        else:
            return  None, 0, 0

    def track_output_to_offsets(self, output):
        """
        This function perform tracking based on the regression deltas.
        :param output: the output of tracking net, which is 3D tensor (or Variable).
                    which has the size of [bs, num_box, 4]. Here, bs is 1, num_box
                    is the number of non-deleted tracks in self.tracks.
        :return:
        """
        # obtain bounding-box regression deltas
        if output is not None:

            if isinstance(output, Variable):
                output = output.data

            box_deltas = output.clone()
            batch_size = box_deltas.size(0)
            if self.cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * self.bbox_reg_std + self.bbox_reg_mean  # [num_box, 4]
                box_deltas = box_deltas.view(batch_size, -1, 4)
            boxes = bbox_transform_inv(boxes=self.boxes.data, deltas=box_deltas,
                                       sigma=self.tracking_model.transform_sigma)  # [1, num_box, 4]

            for t_idx in range(len(self.tracks)):
                self.tracks[t_idx].tracking(bbox_tlbr=boxes[0, t_idx, :])

    def initiate_track(self, detection):
        """
        This function add a track Tracker
        :param detection: detection.Detection
        :return: no return
        """
        self._next_id += 1
        one_track = Track(detection, self._next_id, self.detector_name)
        self.tracks.append(one_track)

    def _match_iou(self, track_candidates, detection_candidates):
        """
        This function match the tracks with detections based on the iou
        :param track_candidates: list, the index of tracks in self.tracks
        :param detection_candidates: list, the index of detections in self.detections_to_track
        :return:
        """
        if len(track_candidates) == 0 or len(detection_candidates) == 0:
            matches = []
            unmatched_tracks = track_candidates
            unmatched_detections = detection_candidates
        else:
            cost_matrix = distance_metric_func.iou_cost(tracks=self.tracks,
                                                        detections=self.detections_to_track,
                                                        track_indices=track_candidates,
                                                        detection_indices=detection_candidates)
            # associate the tracks with the detections using iou
            matches, unmatched_tracks, unmatched_detections = \
                linear_assignment.min_cost_matching_v2(max_distance=self.max_iou_distance,
                                                       tracks=self.tracks,
                                                       detections=self.detections_to_track,
                                                       track_indices=track_candidates,
                                                       detection_indices=detection_candidates,
                                                       cost_matrix=cost_matrix,
                                                       cost_type='iou')

        return matches, unmatched_tracks, unmatched_detections

    def _match_appearance(self, track_candidates, detection_candidates):
        """
        This function match the tracks with detections based on appearance features
        :param track_candidates: list, the index of tracks in self.tracks
        :param detection_candidates: list, the index of detections in self.detections_to_track
        :return:
        """

        if len(track_candidates) == 0 or len(detection_candidates) == 0:
            matches = []
            unmatched_tracks = track_candidates
            unmatched_detections = detection_candidates
        else:

            # compute the euclidean distances between boxes

            dist_cost_matrix = distance_metric_func.euclidean_distance_cost(tracks=self.tracks,
                                                                            detections=self.detections_to_track,
                                                                            track_indices=track_candidates,
                                                                            detection_indices=detection_candidates,
                                                                            image_size=self.im_info[:, 0:2].squeeze())

            # finde the combinations that need to get the appearance similarity
            mask = dist_cost_matrix <= self.max_euclidean_distance
            # 2D tensor, the index of tracks and detection. The first column
            # if the index of tracks, and the second column is the index of
            # detections.  Noted that the index is based on
            # track_candiadtes and detection candidates

            index = torch.nonzero(mask)
            cost_matrix = self.cost_matrix[0:len(track_candidates), 0:len(detection_candidates)].clone()
            if index.size():
                num_pairs = index.size(0)
                # prepare features
                t_feature = None
                d_feature = None
                t_history = [] # used to store the history time of tracks

                for idx_p in range(num_pairs):
                    t_candidate_idx = track_candidates[index[idx_p, 0]]
                    d_candidate_idx = detection_candidates[index[idx_p, 1]]

                    t_feature_tmp = self.tracks[t_candidate_idx].feature  # [history, c, h, w]
                    d_feature_tmp = self.detections_to_track[d_candidate_idx].feature.unsqueeze(dim=0)  # [1, c, h, w]
                    d_feature_tmp = d_feature_tmp.repeat(t_feature_tmp.size(0), 1, 1, 1)  # [history, c, h, w]

                    t_history.append(t_feature_tmp.size(0))

                    if idx_p == 0:
                        t_feature = t_feature_tmp
                        d_feature = d_feature_tmp
                    else:
                        t_feature = torch.cat((t_feature, t_feature_tmp), dim=0)
                        d_feature = torch.cat((d_feature, d_feature_tmp), dim=0)

                # forward to get the appearance similarities
                self.track_feature.data.resize_(t_feature.size()).copy_(t_feature).contiguous()
                self.detection_feature.data.resize_(d_feature.size()).copy_(d_feature).contiguous()

                # prob: [num_f, 2], vis_mask_t: [num_f, h*w, h, w], vis_mask_d: [num_f, h*w, h, w]
                # the memory is limited, so we need to divide to
                bs = self.track_feature.size(0)
                #print('bs: {}'.format(bs))
                max_bs = 10 #3000  # 10000
                num_bs = bs // max_bs
                if num_bs == 0:
                    #t1 = time.time()
                    prob, vis_mask_t, vis_mask_d = self.appearance_model(self.track_feature, self.detection_feature)
                    #t2 = time.time()
                    #print('sbc forward time: {}'.format(t2 - t1))
                else:
                    for i in range(num_bs + 1):
                        start = i * max_bs
                        end = min((i+1)*max_bs, bs)

                        if start != end:

                            prob_tmp, vis_mask_t_tmp, vis_mask_d_tmp = \
                                self.appearance_model(self.track_feature[start:end, :, :, :],
                                                      self.detection_feature[start:end, :, :])

                            if i == 0:
                                prob = prob_tmp.clone()
                                #vis_mask_t = vis_mask_t_tmp.clone()
                                #vis_mask_d = vis_mask_d_tmp.clone()
                            else:
                                prob = torch.cat((prob, prob_tmp), dim=0)
                                #vis_mask_t = torch.cat((vis_mask_t, vis_mask_t_tmp), dim=0)
                                #vis_mask_d = torch.cat((vis_mask_d, vis_mask_d_tmp), dim=0)

                prob = prob.data # 2D, [num_f, 2]
                count = 0
                for idx_p in range(num_pairs):
                    count_next = count + t_history[idx_p]
                    one_prob = prob[count: count_next, 0] # # the probability of been different targets
                    one_prob, _ = one_prob.min(dim=0)
                    row = index[idx_p, 0]
                    col = index[idx_p, 1]
                    cost_matrix[row:row+1, col:col+1] = one_prob
                    count = count_next

            matches, unmatched_tracks, unmatched_detections = \
                linear_assignment.min_cost_matching_v2(max_distance=self.max_appearance_distance,
                                                       tracks=self.tracks,
                                                       detections=self.detections_to_track,
                                                       track_indices=track_candidates,
                                                       detection_indices=detection_candidates,
                                                       cost_matrix=cost_matrix,
                                                       cost_type='appearance')

        return matches, unmatched_tracks, unmatched_detections

    def _match_iou_and_appearance(self, confirmed_tracks, unconfirmed_tracks, unmatched_detections, first_use=None):
        """
        This function match the tracks with detections based on appearance features and iou.
        :param confirmed_tracks: list, the index of confirmed tracks in self.tracks
        :param unconfirmed_tracks: the index of unconfirmed tracks in self.tracks
        :param unmatched_detections: list, the index of detections in self.detections_to_track
        :param first_use: str, 'iou', 'appearance' or 'joint'. The first cost type to use. If
                    'joint', the cost matrix is obtained by the weighted average of iou cost
                    matrix and appearance cost matrix.
        :return:
        """
        if first_use == 'iou':
            # first match the confirmed tracks with detections based on iou,
            # then match the unmatched tracks in confirmed tracks and unconfirmed
            # tracks with the unmatched detections base on appearance.
            track_candidates = confirmed_tracks
            detection_candidates = unmatched_detections
            # t1 = time.time()
            matches_iou, unmatched_tracks_iou, unmatched_detections_iou = \
                self._match_iou(track_candidates=track_candidates,
                                detection_candidates=detection_candidates)
            # t2 = time.time()
            # print('IOU match cost: {}'.format(t2-t1))

            # compute the similarity matrix
            track_candidates = unconfirmed_tracks + unmatched_tracks_iou
            detection_candidates = unmatched_detections_iou

            matches_app, unmatched_tracks_app, unmatched_detections_app = \
                self._match_appearance(track_candidates=track_candidates,
                                       detection_candidates=detection_candidates)

            matches = matches_app + matches_iou

            return matches, unmatched_tracks_app, unmatched_detections_app
        elif first_use == 'appearance':
            # first match the unconfirmed tracks with detections based on appearance,
            # then match the unmatched tracks in unconfirmed tracks and confirmed
            # tracks with the unmatched detections base on appearance.
            track_candidates = confirmed_tracks
            detection_candidates = unmatched_detections

            matches_app, unmatched_tracks_app, unmatched_detections_app = \
                self._match_appearance(track_candidates=track_candidates,
                                       detection_candidates=detection_candidates)

            track_candidates = unconfirmed_tracks + unmatched_tracks_app
            detection_candidates = unmatched_detections_app
            matches_iou, unmatched_tracks_iou, unmatched_detections_iou = \
                self._match_iou(track_candidates=track_candidates,
                                detection_candidates=detection_candidates)

            matches = matches_app + matches_iou

            return matches, unmatched_tracks_iou, unmatched_detections_iou
        elif first_use == 'joint':
            # match the all tracks with detections based on appearance and iou.
            # the cost matrix is obtained by the weighted average of iou cost
            # matrix and appearance cost matrix.

            track_candidates = confirmed_tracks + unconfirmed_tracks
            detection_candidates = unmatched_detections

            if len(track_candidates) == 0 or len(detection_candidates) == 0:
                matches = []
                unmatched_tracks = track_candidates
                unmatched_detections = detection_candidates
            else:
                # obtain the iou cost matrix
                iou_cost_matrix = distance_metric_func.iou_cost(tracks=self.tracks,
                                                                detections=self.detections_to_track,
                                                                track_indices=track_candidates,
                                                                detection_indices=detection_candidates)

                # obtain the appearance cost matrix
                # compute the euclidean distances between boxes
                dist_cost_matrix = distance_metric_func.euclidean_distance_cost(tracks=self.tracks,
                                                                                detections=self.detections_to_track,
                                                                                track_indices=track_candidates,
                                                                                detection_indices=detection_candidates,
                                                                                image_size=self.im_info[:,
                                                                                           0:2].squeeze())

                # finde the combinations that need to get the appearance similarity
                mask = dist_cost_matrix <= self.max_euclidean_distance
                # 2D tensor, the index of tracks and detection. The first column
                # if the index of tracks, and the second column is the index of
                # detections.  Noted that the index is based on
                # track_candiadtes and detection candidates
                index = torch.nonzero(mask)

                # app_cost_matrix = torch.zeros((len(track_candidates), len(detection_candidates))).float().fill_(
                #     linear_assignment.INFTY_COST)
                app_cost_matrix = self.cost_matrix[0:len(track_candidates), 0:len(detection_candidates)].clone()

                if index.size():
                    num_pairs = index.size(0)

                    # prepare features
                    t_feature = None
                    d_feature = None
                    t_history = []  # used to store the history time of tracks
                    for idx_p in range(num_pairs):
                        t_candidate_idx = track_candidates[index[idx_p, 0]]
                        d_candidate_idx = detection_candidates[index[idx_p, 1]]

                        t_feature_tmp = self.tracks[t_candidate_idx].feature  # [num_f, c, h, w]
                        d_feature_tmp = self.detections_to_track[d_candidate_idx].feature.unsqueeze(
                            dim=0)  # [1, c, h, w]
                        d_feature_tmp = d_feature_tmp.repeat(t_feature_tmp.size(0), 1, 1, 1)  # [num_f, c, h, w]

                        t_history.append(t_feature_tmp.size(0))

                        if idx_p == 0:
                            t_feature = t_feature_tmp
                            d_feature = d_feature_tmp
                        else:
                            t_feature = torch.cat((t_feature, t_feature_tmp), dim=0)
                            d_feature = torch.cat((d_feature, d_feature_tmp), dim=0)

                    # forward to get the appearance similarities
                    self.track_feature.data.resize_(t_feature.size()).copy_(t_feature).contiguous()
                    self.detection_feature.data.resize_(d_feature.size()).copy_(d_feature).contiguous()

                    # prob: [num_f, 2], vis_mask_t: [num_f, 1, h, w], vis_mask_d: [num_f, 1, h, w]
                    prob, vis_mask_t, vis_mask_d = self.appearance_model(self.track_feature, self.detection_feature)

                    prob = prob.data  # 2D, [num_f, 2]
                    count = 0

                    for idx_p in range(num_pairs):
                        count_next = count + t_history[idx_p]
                        one_prob = prob[count: count_next, 0]  # # the probability of been different targets
                        one_prob, _ = one_prob.min(dim=0)
                        row = index[idx_p, 0]
                        col = index[idx_p, 1]
                        app_cost_matrix[row:row + 1, col:col + 1] = one_prob
                        count = count_next

                iou_cost_weight = 0.5
                if not iou_cost_matrix.is_cuda and self.args.cuda:
                    iou_cost_matrix = iou_cost_matrix.cuda()
                cost_matrix = iou_cost_weight * iou_cost_matrix + (1 - iou_cost_weight) * app_cost_matrix
                max_distance = iou_cost_weight * self.max_iou_distance + (1 - iou_cost_weight) * self.max_appearance_distance

                matches, unmatched_tracks, unmatched_detections = \
                    linear_assignment.min_cost_matching_v2(max_distance=max_distance,
                                                           tracks=self.tracks,
                                                           detections=self.detections_to_track,
                                                           track_indices=track_candidates,
                                                           detection_indices=detection_candidates,
                                                           cost_matrix=cost_matrix,
                                                           cost_type='joint')

            return matches, unmatched_tracks, unmatched_detections
        else:
            raise RuntimeError('Unknown type of fisrt use: {}'.format(first_use))

    def match(self):
        """
        This function match detections with the tracks.
        :return:
        """

        confirmed_tracks = []
        unconfirmed_tracks = []

        for i, t in enumerate(self.tracks):
            if t.is_confirmed():
                confirmed_tracks.append(i)
            elif t.is_tentative():
                unconfirmed_tracks.append(i)

        unmatched_detections = list(range(len(self.detections_to_track)))

        if self.args.iou_or_appearance == 'appearance':

            track_candidates = confirmed_tracks + unconfirmed_tracks
            detection_candidates = unmatched_detections

            return self._match_appearance(track_candidates=track_candidates,
                                          detection_candidates=detection_candidates)
        elif self.args.iou_or_appearance == 'iou':

            track_candidates = confirmed_tracks + unconfirmed_tracks
            detection_candidates = unmatched_detections

            return self._match_iou(track_candidates=track_candidates, detection_candidates=detection_candidates)

        elif self.args.iou_or_appearance == 'both':

            return self._match_iou_and_appearance(unconfirmed_tracks=unconfirmed_tracks,
                                                  confirmed_tracks=confirmed_tracks,
                                                  unmatched_detections=unmatched_detections,
                                                  #first_use='joint')
                                                  first_use='iou')

    def associated_targets_detections(self):
        """
        This function associate the detections with the tracks
        :return: no return
        """
        matches, unmatched_tracks, unmatched_detections = self.match()

        # update the track set
        for track_idx, detection_idx, cost, distance_type in matches:


            self.tracks[track_idx].update(detection=self.detections_to_track[detection_idx],
                                          cost=cost, distance_type=distance_type)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].predict()

        # initiate tracks
        if self.frame_id == 1 and self._next_id != 0:
            raise ValueError('In the first frame, the number of tracks should be 0 before initialize tracks,'
                             ' but found {}.'.format(self._next_id))
        for detection_idx in unmatched_detections:
            self.initiate_track(self.detections_to_track[detection_idx])

        # filter out those deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def save_tracking_and_detection_results(self):
        """
        This function write the current tracking results
        :param im_scale: scalar, the resized scale for the image
        :return:
        """
        im_scale = self.im_info.data[0, 2]
        for t in self.tracks:
            if t.is_confirmed():
                bbox = t.to_tlwh()
                confidence = t.confidence
                if self.tracking_results is None:
                    self.tracking_results = bbox.new().resize_(0, 10)

                # rescale to origin image
                bbox = bbox / im_scale
                one_data = bbox.new([self.frame_id, t.track_id, bbox[0], bbox[1], bbox[2], bbox[3], confidence, -1, -1, -1]).unsqueeze(dim=0)
                self.tracking_results = torch.cat((self.tracking_results, one_data), dim=0)

        for d in self.detections_to_save:
            bbox = d.to_tlwh()
            feature = d.feature
            len_f = 0
            if feature is not None and self.args.save_detections_with_feature:
                feature = feature.view(1, -1) # 2D
                len_f = feature.size(1)

            confidence = d.confidence
            if self.detection_results is None:
                self.detection_results = bbox.new().resize_(0, 10 + len_f)

            # rescale to origin image
            bbox = bbox / im_scale
            one_data = bbox.new([self.frame_id, -1, bbox[0], bbox[1], bbox[2], bbox[3], confidence, -1, -1, -1]).unsqueeze(dim=0)
            if feature is not None and self.args.save_detections_with_feature:
                one_data = torch.cat((one_data, feature), dim=1) # [1, 10 + len_f]
            self.detection_results = torch.cat((self.detection_results, one_data), dim=0)

    def track_on_video(self, video_file, tracking_output_file, detection_output_file, detector_name):
        """
        Given the path of a video, we do tracking on this video.
        :param video_file: str, the path to a video need to track
        :param tracking_output_file: the file path to save tracking results
        :param detection_output_file: the file path to save detection results
        :return: None
        """
        if not os.path.exists(video_file):
            raise RuntimeError(video_file + ' does not exists')

        video_file_list = video_file.split('/')
        self.seq = video_file_list[-2]
        self.tracked_seqs.append(self.seq)
        self.dataset_year = video_file_list[-4] # MOT16, MOT17
        self.phase = video_file_list[-3]
        self.video_file = video_file
        self.detector_name = detector_name

        base_net_config = 'rcnn_base_i' if self.detector_name == 'PRIVATE' else 'extract_base_features'
        self.base_net_model.set_train_and_test_configure(phase='test', config=base_net_config)

        if self.detector_name != 'PRIVATE':
            det_file = os.path.join(self.mot_dir, self.dataset_year, self.phase, self.seq, 'det', 'det.txt')
            if self.detector_name == 'POI':
                det_file = os.path.join(self.mot_dir, self.dataset_year, 'POI_MOT16_det_feat', self.seq + '_det.txt')
            if self.detector_name == 'DPM':
                det_file = os.path.join(self.mot_dir, self.dataset_year, self.phase, self.seq, 'det', 'det_norm.txt')

            public_detections = pandas.read_csv(det_file).values
            self.public_detections = torch.FloatTensor(public_detections)
            if self.args.cuda:
                self.public_detections = self.public_detections.cuda()

        # infact the num_frames + 1 is the true number of frames in this video
        num_frames = coviar.get_num_frames(video_file) + 1 # in fact, num_frames+1 is the number of frames in this video
        num_gops = coviar.get_num_gops(video_file)

        self.num_frames.append(num_frames)  # add the number of frames for this video

        if num_frames // self.gop_size > num_gops:
            raise RuntimeError('Some thing wrong with the raw video.\n'
                               ' Number of frames: {}, number of GPOs: {}, GOP_SIZE: {}'.
                               format(num_frames, num_gops, self.gop_size))

        # TODO: warm up
        print(' Warming up the tracker...')
        #for frame_id in range(1, self.gop_size + 1):  # frame id starts from 1
        warm_start = 110
        for frame_id in range(warm_start, warm_start + self.gop_size + 1):  # frame id starts from 1
            self.frame_id = frame_id
            print('warming up, frame {}'.format(self.frame_id))
            # The last frame in this video may can not be load (sometimes the
            # im data can not be loaded, sometimes the motion vector and residual)
            # So we will not detect or track on the last frame, just predict the boxes
            if self.frame_id == num_frames:
                pass
            elif (self.frame_id - 1) % self.args.detection_interval == 0:
                # do detection on this frame
                if self.detector_name != 'PRIVATE':
                    _ = self.do_detection_public()
                else:
                    _ = self.do_detection_private()
                self.associated_targets_detections()
            else:
                # do not detect on this frame, just move the boxes of each track to the next frame
                _ = self.do_tracking()
        self.reset()

        # TODO: begin to tracking
        for frame_id in range(1, num_frames + 1):  # frame id starts from 1
            self.frame_id = frame_id

            # The last frame in this video may can not be load (sometimes the
            # im data can not be loaded, sometimes the motion vector and residual)
            # So we will not detect or track on the last frame, just predict the boxes
            if self.frame_id == num_frames:
                print('predict on {}, number of frames: {}/{}, number of tracks: {}. Detector: {}'.
                      format(video_file, frame_id, num_frames, len(self.tracks), self.detector_name))
                t1 = time.time()
                for t in self.tracks:
                    t.predict()

                t2 = time.time()
                # we treat the time consuming as the offseting time consumption
                self.offset_time.append(t2 - t1)
                # save results
                self.save_tracking_and_detection_results()

            elif (self.frame_id - 1) % self.args.detection_interval == 0:
                # for those frames that need to be detected, do detection
                print('detect on {}, number of frames: {}/{}, number of tracks: {}. Detector: {}'.
                      format(video_file, frame_id, num_frames, len(self.tracks), self.detector_name))
                # do detection on this frame
                if self.detector_name != 'PRIVATE':
                    load_time, detect_time = self.do_detection_public()
                else:
                    load_time, detect_time = self.do_detection_private()

                self.load_time.append(load_time)
                self.detect_time.append(detect_time)

                association_start = time.time()
                self.associated_targets_detections()
                associate_time = time.time() - association_start

                self.associate_time.append(associate_time)

                self.save_tracking_and_detection_results()
            else:
                print('track on {}, number of frames: {}/{}, number of tracks {}. Detector: {}'.
                      format(video_file, frame_id, num_frames, len(self.tracks), self.detector_name))
                # do not detect on this frame, just move the boxes of each track to the next frame
                output, load_time, track_time = self.do_tracking()

                t1 = time.time()
                self.track_output_to_offsets(output)
                offset_time = time.time() - t1

                self.save_tracking_and_detection_results()

                # save time
                self.load_time.append(load_time)
                self.track_time.append(track_time)
                self.offset_time.append(offset_time)

            if self.args.vis:
                self.visualize_results(kind_of_boxes='tracking')

        # reset the tracker so it can track on the next video
        self.reset(tracking_output_file=tracking_output_file, detection_output_file=detection_output_file)


