
import numpy as np
import torch
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import cv2
import coviar
import os
from lib.utils.image_viewer import ImageViewer
import colorsys

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]
        self.frame_interval = seq_info["frame_interval"]

    def set_image(self, image, frame_id=None):
        pass

    # def draw_detections(self, detections):
    #     pass
    #
    # def draw_trackers(self, trackers):
    #     pass

    def draw_box(self, box):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking results in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms):
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]
        self.frame_interval = seq_info["frame_interval"]

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += self.frame_interval
        return True

    def set_image(self, image, frame_id=None, color=None):
        """
        This function set the show the image and id number
        :param image: the image, bgr
        :param frame_id: scalar, the id of this image
        :param color: tuple with length 3, the color of frame_id to show, (b, g, r)
        :return:
        """
        self.viewer.image = image
        color = (199, 255, 84) if color is None else color
        if frame_id is not None:
            if not isinstance(frame_id, str):
                frame_id = str(frame_id)
            frame_id = '#' + frame_id
            self.viewer.annotate(20, 70, frame_id, color)


    # def draw_detections(self, detections):
    #     self.viewer.thickness = 2
    #     self.viewer.color = 0, 0, 255
    #     for i, detection in enumerate(detections):
    #         tlwh = np.asarray(detection.tlwh)
    #         x, y, w, h = tlwh[0], tlwh[1], tlwh[2], tlwh[3]
    #
    #         self.viewer.rectangle(x, y, w, h)
    #
    # def draw_trackers(self, tracks):
    #     self.viewer.thickness = 2
    #     for track in tracks:
    #         if not track.is_confirmed() or track.time_since_update > 0:
    #             continue
    #         self.viewer.color = create_unique_color_uchar(track.track_id)
    #         tlwh = np.asarray(track.to_tlwh(), dtype=np.int)
    #         x, y, w, h = tlwh[0], tlwh[1], tlwh[2], tlwh[3]
    #
    #         self.viewer.rectangle(x, y, w, h, label=str(track.track_id))
    #         # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
    #         #                      label="%d" % track.track_id)

    def draw_box(self, boxes, box_type='mot'):
        """
        This function show boxes on the viewer
        :param boxes: a list, each element in it is a ndarray with size (10,), and it has the same format with
                    MOTChallenge det or gt file
        :param box_type: str, can be ['mot', 'tlbr', 'tlwh']
        :return: None
        """
        boxes = np.asarray(boxes)
        self.viewer.thickness = 2
        for box in boxes:
            target_id = int(box[0])
            tlwh = box[1: 5]

            if box_type == 'tlbr':
                tlwh[2:4] = tlwh[2:4] - tlwh[0:2]
            if len(box) > 5:
                #score = str(box[5])
                score = None
            else:
                score = None

            if target_id <= 0: # detection
                # self.viewer.color = create_unique_color_uchar(random.randint(-100, 100))
                self.viewer.color = 0, 0, 255
                self.viewer.rectangle(x=tlwh[0], y=tlwh[1], w=tlwh[2], h=tlwh[3], label_br=score)
            else: # gt or track results
                self.viewer.color = create_unique_color_uchar(target_id)
                self.viewer.rectangle(x=tlwh[0], y=tlwh[1], w=tlwh[2], h=tlwh[3], label_tl=str(target_id), label_br=score)


def show_feature_map(in_feature, batch_idx=0, min_channel=-1, max_channel=-1,
                     save=False, save_path='images/feature_map.pdf',
                     show=True, show_size=None):
    """
    This function visulaize the feature map, so we can
    analyse it directly.
    :param in_feature: Variable or Tensor, 4D, [bs x C x H x W]
    :param batch_idx: scalar, the batch index to show
    :param save: bool, whether to save the showed results
    :param save_path: the path to save the figure
    :param show: bool, whether to show this figure
    :param show_size: tuple, (w, h), the size to show each channel feature map.
                If None, the feature map will be shown in its origin size.
    :return: None
    """

    # in_feature = in_feature.view(in_feature.size(0), in_feature.size(1), -1) # [bs, c, hxw]
    # in_feature = in_feature.permute(0, 2, 1).contiguous()
    # in_feature = in_feature.view(in_feature.size(0), in_feature.size(1), 7, 7)


    if isinstance(in_feature, Variable):
        in_feature = in_feature.data
    #     if in_feature.is_cuda:
    #         in_feature = in_feature.cpu()
    #     in_feature = in_feature.cpu()
    #
    # if isinstance(in_feature, torch.FloatTensor) or isinstance(in_feature, torch.Tensor):
    #     in_feature = in_feature.numpy()
    in_feature = np.asarray(in_feature)
    size = in_feature.shape

    if len(size) < 4:
        raise RuntimeError('Expected input feature is 4D, but we get {}D input feature'.format(len(size)))

    bs, c, h, w = size[0], size[1], size[2], size[3]
    min_channel = min_channel if (min_channel >= 0 and min_channel <= c) else 0
    max_channel = max_channel if (max_channel >=0 and max_channel <= c) else c
    if min_channel > max_channel:
        raise RuntimeError('Invalid min_channel {} and max_channel {}'.format(min_channel, max_channel))

    show_feature = in_feature[batch_idx]
    # show_feature = show_feature / np.max(show_feature) * 255
    # show_feature = show_feature / 1000 * 255
    show_feature = show_feature * 255
    grid_w = math.ceil(math.sqrt(max_channel-min_channel))
    grid_h = math.ceil((max_channel - min_channel) / float(grid_w))

    plt.figure(figsize=(grid_h, grid_w))
    plt.tight_layout()

    for c_idx in range(min_channel, max_channel):
        print('Ploting {} channel...'.format(c_idx - min_channel))
        feat_map = show_feature[c_idx]
        feat_map = np.array(feat_map, dtype=np.uint8)
        if show_size is not None:
            feat_map = cv2.resize(feat_map, show_size)
        plt.subplot(grid_h, grid_w, c_idx+1)
        plt.imshow(feat_map)
        plt.axis('off')

    if save:
        plt.savefig(save_path)

    if show:
        plt.show()

    print('Show feature map done! ')


def compressed_frame_to_show(frame, frame_type, tool_type='plt'):
    """
    This function convert the frame extracted from compressed video to rgb image.
     :param frame: ndarray, the frame extracted from the compressed video, there are three type of frames, which is
                figured out by the frame_type. It should be noted that the frame is the raw data extracted by the
                function coviar.load() with the format BGR
    :param frame_type: scalar, which can be 0, 1 or 2. 0 for I frame, 1 for motion vector, 2 for residuals
    :param tool_type: string, 'plt' or 'cv2', the tools used to show
    :return: RGB image
    """
    if frame_type not in [0, 1, 2]:
        raise RuntimeError('Invalid frame type, expected is 0, 1 or 2.')
    else:
        if frame_type == 0:  # the I frame, i.e. the image, frame is 3D ndarray [w, h, 3]
            # plt show image in the format RGB, while cv2 show image in the format BGR
            if tool_type == 'plt':
                frame = frame[:, :, [2, 1, 0]]  # from BGR to RGB, uncomment this for plt
            elif tool_type == 'cv2':
                pass # for cv2

        elif frame_type == 1:  # motion vector
            s = np.shape(frame)
            new_s = (s[0], s[1], 3)
            hsv = np.zeros(new_s)  # make a hsv motion vector map

            # get the direction, we show the direction in the h channel
            tan = frame[:, :, 0] / (frame[:, :, 1] + 1e-10)
            hsv[:, :, 0] = np.arctan(tan)
            hsv[:, :, 0] = hsv[:, :, 0] - np.min(hsv[:, :, 0])
            hsv[:, :, 0] = hsv[:, :, 0] / np.pi * 179

            # get the amplitude, we show amplitude in the s channel
            hsv[:, :, 1] = np.sqrt(np.power(frame[:, :, 0], 2) + np.power(frame[:, :, 1], 2))
            hsv[:, :, 1] = hsv[:, :, 1] / np.max(hsv[:, :, 1]) * 255

            # make the v channel
            hsv[:, :, 2] = 255

            hsv = hsv.astype(np.uint8)  # change to uint8
            if tool_type == 'plt':
                hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) # for plt
            elif tool_type == 'cv2':
                hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # for cv2

            frame = hsv

        elif frame_type == 2:  # residual, frame is 3D ndarray [w, h, 3]

            if tool_type == 'plt':
                frame = frame[:, :, [2, 1, 0]]  # from BGR to RGB, uncoment this for plt

            frame = np.abs(frame)  # absolute value since the residual can be negative
            # the background is usually static, so in residual it will be black,
            # we minus it by 255 to change it to white
            frame = 255 - frame
            frame = frame.astype(np.uint8)  # change to uint8

        return frame


def show_compressed_frame(frame, frame_type, title='', tool_type='plt', show=True, save=False, path='/results.pdf'):
    """
    this function show the frames in the compressed video
    :param frame: ndarray, the frame extracted from the compressed video, there are three type of frames, which is
                figured out by the frame_type. It should be noted that the frame is the raw data extracted by the
                function coviar.load()
    :param frame_type: scalar, which can be 0, 1 or 2. 0 for I frame (RGB images), 1 for motion vector, 2 for residuals
    :return: None
    """
    frame = compressed_frame_to_show(frame, frame_type, tool_type=tool_type)
    if tool_type == 'cv2':
        cv2.imshow('', frame)
        cv2.waitKey(0)
    elif tool_type == 'plt':
        plt.figure()
        plt.imshow(frame)
        plt.axis('off')
        plt.title(title)
        if save:
            plt.savefig(path)
        if show:
            plt.show()


def show_boxes_in_compressed_video(video_path, update_ms=10, min_confidence=0.0, box_file_path = None,
                                min_frame_idx=None, max_frame_idx=None, frame_interval=1, frame_type=0, accumulate=False):
    """
    This function show box

    :param box_file_path: string, the path of the boxes. The format of this file should be the same with MOTChallenge
                det.txt or gt.txt.
    :param video_path: string, the path of frames.
    :param update_ms: scalar, 1000 / update_ms is the fps, default 10
    :param min_confidence: float, the confidence threshold of detection, the boxes with smaller confidence will not be
                displayed. Default 0.0
    :param min_frame_idx: integer, the first frame to display, default the first frame of this sequence
    :param max_frame_idx: integer, the last frame to display, default the last frame of this sequence
    :param frame_interval: the interval to show frames
    :param frame_type: int, can be 0, 1 or 2 (denotes I frame, motion vector, residual, respectively)
    :param accumulate: used for the motion vector and residual. If it is true, the motion vector and residual are
                accumulated.
    :return: None
    """

    def frame_callback(vis, frame_idx):
        #
        print('Processing frame: ', frame_idx)

        # Load image and generate detections.
        # Update visualization.
        GROUP_SIZE = 12 # the number of frames in one group. We set to 12 for the raw mpeg4 video.
        gop_idx = int((frame_idx - 1) / GROUP_SIZE) # GOP starts from 0, while frame_idx here starts from 1.
        in_group_idx = int((frame_idx - 1) % GROUP_SIZE) # the index in the group
        image = coviar.load(video_path, gop_idx, in_group_idx, frame_type, accumulate)
        image = compressed_frame_to_show(image, frame_type, tool_type='cv2')

        vis.set_image(image.copy(), frame_idx)

        raw_box = seq_info['boxes']
        if raw_boxes is not None:
            index = raw_box[:, 0] == frame_idx
            box = raw_box[index]
            index = box[:, 6] >= min_confidence
            box = box[index]
            box = box[:, 1:7]  # [target_id, x, y, w, h]
            box_list = []
            for idx in range(box.shape[0]):
                box_list.append(box[idx, :])
            vis.draw_box(box_list)

    total_frames = coviar.get_num_frames(video_path) + 1

    # get the first and las frame index
    if min_frame_idx is None:
        min_frame_idx = 1
    if min_frame_idx < 0 or min_frame_idx > total_frames:
        min_frame_idx = 1

    if max_frame_idx is None:
        max_frame_idx = total_frames
    if max_frame_idx < 0 or max_frame_idx > total_frames:
        max_frame_idx = total_frames

    if min_frame_idx > max_frame_idx:
        raise RuntimeError('The first frame index ', min_frame_idx, ' is larger than the last frame index ',
                           max_frame_idx)

    # get the sequence information
    im = coviar.load(video_path, 0, 0, 0, False)
    im_size = im.shape

    raw_boxes = None if box_file_path is None else np.loadtxt(box_file_path, dtype=float, delimiter=',')

    seq_info = {
        'image_size': [im_size[0], im_size[1]],
        'min_frame_idx': min_frame_idx,
        'max_frame_idx': max_frame_idx,
        'frame_interval': frame_interval,
        'boxes': raw_boxes,
        'sequence_name': ''
    }

    visualizer = Visualization(seq_info, update_ms)
    visualizer.run(frame_callback)


def show_boxes_in_separate_frames(frame_path, update_ms=10, min_confidence=0.0, box_file_path=None,
                                  min_frame_idx=None, max_frame_idx=None, frame_interval=1):
    """
    this function show boxes in the video

    :param box_file_path: string, the path of the boxes. The format of this file should be the same with MOTChallenge
                det.txt or gt.txt.
    :param frame_path: string, the path of frames.
    :param update_ms: scalar, 1000 / update_ms is the fps, default 10
    :param min_confidence: float, the confidence threshold of detection, the boxes with smaller confidence will not be
                displayed. Default 0.0
    :param min_frame_idx: integer, the first frame to display, default the first frame of this sequence
    :param max_frame_idx: integer, the last frame to display, default the last frame of this sequence
    :param frame_interval: the interval to show frames
    :return: None
    """

    def frame_callback(vis, frame_idx):
        print('Processing frame: ', frame_idx)

        # Load image and generate detections.
        im_name = str(frame_idx).zfill(6) + '.jpg'
        image = cv2.imread(os.path.join(frame_path, im_name), cv2.IMREAD_COLOR)

        vis.set_image(image.copy(), frame_idx)
        raw_box = seq_info['boxes']
        if raw_boxes is not None:

            index = raw_box[:, 0] == frame_idx
            box = raw_box[index]
            index = box[:, 6] >= min_confidence
            box = box[index]
            box = box[:, 1:7] # [target_id, x, y, w, h, score]

            # index = raw_box[:, 0] <= frame_idx
            # box = raw_box[index]
            # index = box[:, 0] > frame_idx - 12
            # box = box[index]
            # index = box[:, 6] > min_confidence
            # box = box[index]
            # box = box[:, 1:7]



            box_list = []
            for idx in range(box.shape[0]):
                box_list.append(box[idx, :])
            vis.draw_box(box_list)

    im_file_names = os.listdir(frame_path)
    im_file_names.sort()
    total_frames = len(im_file_names)

    # get the first and las frame index
    if min_frame_idx is None:
        min_frame_idx = 1
    if min_frame_idx < 0 or min_frame_idx > total_frames:
        min_frame_idx = 1

    if max_frame_idx is None:
        max_frame_idx = total_frames
    if max_frame_idx < 0 or max_frame_idx > total_frames:
        max_frame_idx = total_frames

    if min_frame_idx > max_frame_idx:
        raise RuntimeError('The first frame index ', min_frame_idx, ' is larger than the last frame index ', max_frame_idx)

    # get the sequence information
    im_path = os.path.join(frame_path, im_file_names[0])
    im = cv2.imread(im_path)
    im_size = im.shape

    raw_boxes = None if box_file_path is None else np.loadtxt(box_file_path, dtype=float, delimiter=',')

    seq_info = {
        'image_size': [im_size[0], im_size[1]],
        'min_frame_idx': min_frame_idx,
        'max_frame_idx': max_frame_idx,
        'frame_interval': frame_interval,
        'boxes': raw_boxes,
        'sequence_name': ''
    }

    visualizer = Visualization(seq_info, update_ms)
    visualizer.run(frame_callback)
