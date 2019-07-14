import numpy as np
import json

def filter_mot_gt_boxes(gt_boxes, vis_threshold=0.1, ambiguous_class_id=None):
    """
    This function filter the gt boxes of mot17 and 16
    :param gt_boxes: array_like, the raw data load from file gt.txt
    :param vis_threshold: the threshold of visibility, lower will be filtered out
    :param ambiguous_class_id: if ambiguous is None, we will not preserve the ambiguous class
                else, we will preserve those boxes
    :return: filtered gt_boxes
    """

    class_id = {'Pedestrain': 1,
                'person_on_vehicle': 2,
                'Car': 3,
                'Bicycle': 4,
                'Motorbike': 5,
                'Non_motorized_vehicle': 6,
                'Static_person': 7,
                'Distractor': 8,
                'Occluder': 9,
                'Occluder_on_the_ground': 10,
                'Occluder_full': 11,
                'Reflection': 12}

    # we set the class_id 1, 2, 7 as positive, 8 as ambiguous, and the other as negative.
    # So we only need to preserve the 1, 2, 7, 8 classes
    idx1 = gt_boxes[:, 7] == 1
    idx2 = gt_boxes[:, 7] == 2
    idx7 = gt_boxes[:, 7] == 7
    idx = idx1 + idx2 + idx7
    if ambiguous_class_id is not None:
        idx8 = gt_boxes[:, 7] == 8
        idx = idx + idx8
    gt_boxes = gt_boxes[idx]

    # the 8th (starts from 0) denotes the visibility of this object, we use 0.1
    # to filter those boxes have low visibility
    idx = gt_boxes[:, 8] >= vis_threshold
    gt_boxes = gt_boxes[idx]

    return gt_boxes


def get_one_tracklet(gt_boxes, length, type, target_id=None, interval=1, shuffle=False, jitter=False, iou_thr=0.7, im_shape=None):
    """
    This function generate a tracklet
    :param gt_boxes: the gt boxes loaded from the gt.txt
    :param length: the number of frames of tracklet to generate
    :param type: string, 'positive' or 'negative'
    :param target_id: integer, the id of target. If None, choose a target randomly.
    :param interval: integer, we sample a detection every 'interval' frames.
    :param shuffle: bool, whether to shuffle the tracklets.
    :param jitter: bool, whether to jitter box
    :param iou_thr: the boxes obtained from the origin boxes will be kept if iou > iou_thr
    :param im_shape: array, [h, w, c]. if None, we will not clip the box.
    :return: array, the tracklet, shape[0] is the length of tracklet, each row has the same format as follows:
                for MOT17: [frame_id, target_id, x1, y1, x2, y2, consideration, category_id, visibility]
                for MOT15: [frame_id, target_id, x1, y1, x2, y2, consideration, -1, -1, -1]
    """

    target_ids = np.unique(gt_boxes[:, 1])
    if target_id is not None and target_id not in target_ids:
        raise ValueError('Invalid target id: {}'.format(target_id))

    if target_id is None:
        random_target_id = True
    else:
        random_target_id = False

    done = False
    while not done:
        if random_target_id:
            target_id = np.random.choice(target_ids)
        #print('Generate the tracklet for target {}'.format(target_id))

        index = gt_boxes[:, 1] == target_id
        positive_gt_boxes = gt_boxes[index]

        index = gt_boxes[:, 1] != target_id
        negative_gt_boxes = gt_boxes[index]

        life = positive_gt_boxes.shape[0]

        # choose the start frame randomly
        if type == 'positive':
            if life < length * interval:
                # in this case, we set interval==1
                if life > length:
                    start = np.random.choice(range(life - (length - 1)))
                    positive_index = np.array(range(start, start + length))
                    done = True
                elif life == length:
                    # choose all boxes in this tracklet
                    positive_index = np.array(range(life))
                    done = True
                else:
                    if random_target_id:
                        continue
                    else:
                        raise RuntimeError('The life of this target {} is less the required length of tracklet {}'.format(life, length))
            else:
                start = np.random.choice(range(positive_gt_boxes.shape[0] - (length - 1) * interval))
                positive_index = np.array(range(start, start + length * interval, interval))
                done = True

            tracklet = positive_gt_boxes[positive_index]
            if shuffle:
                np.random.shuffle(tracklet)
        else:
            if life < (length-1) * interval:
                # in this case, we set interval==1
                if life > length - 1:
                    start = np.random.choice(range(life - (length - 1 - 1)))
                    positive_index = np.array(range(start, start + length - 1))
                    done = True
                elif life == length-1:
                    # choose all boxes in this tracklet
                    positive_index = np.array(range(life))
                    done = True
                else:
                    if random_target_id:
                        continue
                    else:
                        raise RuntimeError('The life of this target {} is less the required length of tracklet {}'.format(life, length))
            else:
                start = np.random.choice(range(positive_gt_boxes.shape[0] - (length - 2) * interval))
                positive_index = np.array(range(start, start + (length-1) * interval, interval))
                done = True
            tracklet = positive_gt_boxes[positive_index]
            if shuffle:
                np.random.shuffle(tracklet)
            # append one negative box
            one_negative = negative_gt_boxes[np.random.choice(range(negative_gt_boxes.shape[0]))]
            one_negative = one_negative[np.newaxis, :]
            tracklet = np.concatenate((tracklet, one_negative), axis=0)

    if tracklet.shape[0] != length:
        raise RuntimeError('The length of generated tracklet {} is less than the required length {}'.format(tracklet.shape[0], length))

    if jitter:
        tracklet = jitter_tracklets(tracklets=tracklet, iou_thr=iou_thr, up_or_low='up')

    # clip boxes to the image boundary
    bbox = tracklet[:, 2:6]  # x1, y1, w, h
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    w = bbox[:, 2]
    h = bbox[:, 3]
    x2 = x1 + w
    y2 = y1 + h
    refined_bbox = bbox.copy()
    if im_shape is not None:
        refined_bbox[:, 0] = x1.clip(0, im_shape[1] - 1)
        refined_bbox[:, 1] = y1.clip(0, im_shape[0] - 1)
        refined_bbox[:, 2] = x2.clip(0, im_shape[1] - 1)
        refined_bbox[:, 3] = y2.clip(0, im_shape[0] - 1)
    else:
        refined_bbox[:, 0] = x1
        refined_bbox[:, 1] = y1
        refined_bbox[:, 2] = x2
        refined_bbox[:, 3] = y2
    tracklet[:, 2:6] = refined_bbox

    return tracklet


def get_one_patch_pair(gt_boxes, length, type, target_id=None, jitter=False, iou_thr=0.7, im_shape=None):
    """
    This function generate a pair of image (target) patch. We sample two image patch randomly. The two patch may from
    the same frame, maybe from two different frames.
    :param gt_boxes: the gt boxes loaded from the gt.txt
    :param length: the number of frames of tracklet to generate
    :param type: string, 'positive' or 'negative'
    :param target_id: integer, the id of target. If None, choose a target randomly.
    :param jitter: bool, whether to jitter box
    :param iou_thr: the boxes obtained from the origin boxes will be kept if iou > iou_thr
    :param im_shape: array, [h, w, c]. if None, we will not clip the box.
    :return: array, the tracklet, shape[0] is the length of tracklet, each row has the same format as follows:
                for MOT17: [frame_id, target_id, x1, y1, x2, y2, consideration, category_id, visibility]
                for MOT15: [frame_id, target_id, x1, y1, x2, y2, consideration, -1, -1, -1]
    """

    target_ids = np.unique(gt_boxes[:, 1])
    if target_id is not None and target_id not in target_ids:
        raise ValueError('Invalid target id: {}'.format(target_id))

    if length != 2:
        raise ValueError('In valid length, expected is 2, but got {}'.format(length))

    if target_id is None:
        random_target_id = True
    else:
        random_target_id = False

    done = False
    while not done:
        if random_target_id:
            target_id = np.random.choice(target_ids)
        #print('Generate the tracklet for target {}'.format(target_id))

        index = gt_boxes[:, 1] == target_id
        positive_gt_boxes = gt_boxes[index]

        index = gt_boxes[:, 1] != target_id
        negative_gt_boxes = gt_boxes[index]

        life = positive_gt_boxes.shape[0]

        if life == 1 and random_target_id:
            continue
        elif life == 1 and not random_target_id:
            raise RuntimeError('The life of target {} is 1'.format(target_id))

        # choose the start frame randomly
        if type == 'positive':
            index1 = np.random.choice(list(range(positive_gt_boxes.shape[0])))
            tracklet1 = positive_gt_boxes[index1] # [10]
            tracklet1 = tracklet1[np.newaxis, :]

            if jitter:
                tracklet1 = jitter_tracklets(tracklets=tracklet1, iou_thr=iou_thr, up_or_low='up')

            # append another positive box
            choice = np.random.choice([0, 1])
            if choice == 0:
                # choose anther box from other frame
                index2 = np.random.choice(list(range(index1)) + list(range(index1 + 1, positive_gt_boxes.shape[0])))
                tracklet2 = positive_gt_boxes[index2]  # [1, 10]
                tracklet2 = tracklet2[np.newaxis, :]

                if jitter:
                    tracklet2 = jitter_tracklets(tracklets=tracklet2, iou_thr=iou_thr, up_or_low='up')
            else:
                # generate the positive from it self
                tracklet2 = jitter_tracklets(tracklets=tracklet1.copy(), iou_thr=iou_thr, up_or_low='up')

            tracklet = np.concatenate((tracklet1, tracklet2), axis=0)
            done = True

        else:
            index1 = np.random.choice(list(range(positive_gt_boxes.shape[0])))
            tracklet_positive = positive_gt_boxes[index1] # [10]
            tracklet_positive = tracklet_positive[np.newaxis, :] # [1, 10]
            if jitter:
                tracklet_positive = jitter_tracklets(tracklets=tracklet_positive, iou_thr=iou_thr, up_or_low='up')

            # append one negative box
            choice = np.random.choice([0, 1])
            if choice == 0:
                # choose one other target as negative
                tracklet_negative = negative_gt_boxes[np.random.choice(range(negative_gt_boxes.shape[0]))]
                tracklet_negative = tracklet_negative[np.newaxis, :]
                if jitter:
                    tracklet_negative = jitter_tracklets(tracklets=tracklet_negative, iou_thr=iou_thr, up_or_low='up')
            else:
                # generate the negative from it self
                tracklet_negative = jitter_tracklets(tracklets=tracklet_positive.copy(), iou_thr=0.3, up_or_low='low')

            tracklet = np.concatenate((tracklet_positive, tracklet_negative), axis=0)
            done = True

    if tracklet.shape[0] != length:
        raise RuntimeError('The length of generated tracklet {} is less than the required length {}'.format(tracklet.shape[0], length))

    # clip boxes to the image boundary
    bbox = tracklet[:, 2:6]  # x1, y1, w, h
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    w = bbox[:, 2]
    h = bbox[:, 3]
    x2 = x1 + w
    y2 = y1 + h
    refined_bbox = bbox.copy()
    if im_shape is not None:
        refined_bbox[:, 0] = x1.clip(0, im_shape[1] - 1)
        refined_bbox[:, 1] = y1.clip(0, im_shape[0] - 1)
        refined_bbox[:, 2] = x2.clip(0, im_shape[1] - 1)
        refined_bbox[:, 3] = y2.clip(0, im_shape[0] - 1)
    else:
        refined_bbox[:, 0] = x1
        refined_bbox[:, 1] = y1
        refined_bbox[:, 2] = x2
        refined_bbox[:, 3] = y2
    tracklet[:, 2:6] = refined_bbox

    return tracklet


def jitter_tracklets(tracklets, iou_thr=0.7, up_or_low=None, format='mot'):
    """
    This function generate a box from the origin box, i.e. jitter the
    origin box.
    :param tracklets: 2D array
    :param iou_thr: the boxes obtained from the origin boxes will be kept if iou > iou_thr
    :param up_or_low: string, 'up' or 'low'
    :param format: str, the format of tracklets. 'mot' means it has the same format with motchallenge.
                for MOT17: [frame_id, target_id, x1, y1, w, h, consideration, category_id, visibility]
                for MOT15: [frame_id, target_id, x1, y1, w, h, consideration, -1, -1, -1].

    :return: jittered tracklets, has the same size with tracklets.
    """
    # for MOT17: [frame_id, target_id, x1, y1, w, h, consideration, category_id, visibility]
    # for MOT17: [frame_id, target_id, x1, y1, w, h, consideration, -1, -1, -1]

    if format == 'mot':
        jitter_t = tracklets.copy()
        boxes = tracklets[:, 2:6]
        num_boxes = boxes.shape[0]

        # jitter each box
        for i in range(num_boxes):
            one_box = boxes[i]
            one_box_jitter = jitter_a_box(one_box=one_box, iou_thr=iou_thr, up_or_low=up_or_low)
            jitter_t[i, 2:6] = one_box_jitter
    elif format == 'xywh': # each row in it is [x1, y1, w, h]
        jitter_t = tracklets.copy()
        num_boxes = jitter_t.shape[0]

        # jitter each box
        for i in range(num_boxes):
            one_box = tracklets[i]
            one_box_jitter = jitter_a_box(one_box=one_box, iou_thr=iou_thr, up_or_low=up_or_low)
            jitter_t[i, :] = one_box_jitter
    elif format == 'tlbr': # each row in it is [x1, y1, x2, y2]
        w = tracklets[:, 2] - tracklets[:, 0] + 1
        h = tracklets[:, 3] - tracklets[:, 1] + 1
        tracklets[:, 2] = w
        tracklets[:, 3] = h

        jitter_t = tracklets.copy()
        num_boxes = jitter_t.shape[0]

        # jitter each box
        for i in range(num_boxes):
            one_box = tracklets[i]
            one_box_jitter = jitter_a_box(one_box=one_box, iou_thr=iou_thr, up_or_low=up_or_low)
            jitter_t[i, :] = one_box_jitter

        x2 = jitter_t[:, 0] + jitter_t[:, 2] - 1
        y2 = jitter_t[:, 1] + jitter_t[:, 3] - 1
        jitter_t[:, 2] = x2
        jitter_t[:, 3] = y2
    else:
        raise NotImplementedError

    return jitter_t


def jitter_a_box(one_box, iou_thr=None, up_or_low=None):
    """
    This function jitter a box
    :param box: [x1, y1, w, h]
    :param iou_thr: the overlap threshold
    :param up_or_low: string, 'up' or 'low'
    :return:
    """

    # get the std
    # 1.96 is the interval of probability 95% (i.e. 0.5 * (1 + erf(1.96/sqrt(2))) = 0.975)
    std_xy = one_box[2: 4] / (2 * 1.96)
    std_wh = 10 * np.tanh(np.log10(one_box[2:4]))
    std = np.concatenate((std_xy, std_wh), axis=0)

    if up_or_low == 'up':
        index = np.array([])
        while index.shape[0] <= 0:
            jitter_boxes = np.random.normal(loc=one_box, scale=std, size=(100, 4))
            overlap = iou(one_box, jitter_boxes)
            index = overlap >= iou_thr
            index = np.nonzero(index)[0]
    elif up_or_low == 'low':
        index = np.array([])
        while index.shape[0] <= 0:
            jitter_boxes = np.random.normal(loc=one_box, scale=std, size=(100, 4))
            overlap = iou(one_box, jitter_boxes)
            index = (overlap <= iou_thr) & (overlap >= 0)
            index = np.nonzero(index)[0]

    choose_index = index[np.random.choice(range(index.shape[0]))]
    choose_box = jitter_boxes[choose_index]

    return choose_box


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def load_citypersons_annotation(ann_path, im_info_path, frame_id, ambiguous_class_id, vis_threshold=0.1):
    """
    This function load the annotation of one image.
    :param ann_path: str, the path to the annotation json file path
    :param ambiguous_class_id: int, the id denote that this class is 'do not care'. If None, the
            ambiguous boxes will not be loaded, and it will be treated as bg.
    :param vis_threshold: scalar, the visibility that lower than this
            threshold will be discarded
    :param im_info_path: path to im_info file, the information of the compressed short video.
            each line in it is [frame_id, x1, y1, x2, y2, x_scale, y_scale]. Please refer to
            scripts citypersons_images2video.py for more information.
    :return: a dictionary
    """
    category_id = {'ignore': 0,
                   'pedestrian': 1,
                   'rider': 2,
                   'sitting person': 3,
                   'person (other)': 4,
                   'person group': 5 # note that the bboxVis of person group are all zeros !
                   }

    with open(ann_path, 'r') as ann_f:
        ann = json.load(ann_f)

    im_width = ann['imgWidth']
    im_height = ann['imgHeight']

    objects = ann['objects']  # a list
    bbox = np.zeros((len(objects), 4))  # [x1, y1, w, h, vis]
    visibility = np.zeros(len(objects))
    gt_class = np.zeros(len(objects))  # used to preserve the
    for idx in range(len(objects)):
        one_obj = objects[idx]
        one_bbox = np.array(one_obj['bbox'])
        one_bbox_vis = np.array(one_obj['bboxVis'])
        one_visibility = (one_bbox_vis[2] * one_bbox_vis[3]) / (one_bbox[2] * one_bbox[3])
        one_category = one_obj['label']
        if one_category == 'person group':
            # the bboxVis of this category are all zeros, hence we set it to 1
            one_visibility = 1

        bbox[idx,:] = one_bbox
        visibility[idx] = one_visibility
        # class id
        cat_id = category_id[one_category]
        if cat_id in [1, 2, 3, 4, 5]:
            class_id = 1
        elif cat_id in [0]: # bg
            class_id = 0
        gt_class[idx] = class_id

    # filter out bg
    idx = gt_class != 0
    gt_class = gt_class[idx]
    bbox = bbox[idx]
    visibility = visibility[idx]

    # filter out lower visibility
    idx = visibility >= vis_threshold
    bbox = bbox[idx]
    gt_class = gt_class[idx]

    # # filter out small boxes
    # area = bbox[:, 2] * bbox[:, 3] / (im_height * im_width)
    # idx_area = area >= 1 / 20000. # the min area ration on mot is about 65e-6

    ratio = bbox[:, 3] / bbox[:, 2] # h / w
    idx1 = ratio >= 0.8 # # the min area ration on mot is about 0.9047619
    idx2 = ratio <= 8
    idx = idx1 & idx2
    bbox = bbox[idx]
    gt_class = gt_class[idx]

    # modify the box according to the im_info
    im_info = np.loadtxt(im_info_path, delimiter=',')
    # pick out the corresponding im_info
    im_info = im_info[int(frame_id - 1)]
    # [frame_id, x1, y1, x2, y2, x_scale, y_scale]
    # x1, x2, y1, y2 are the coordinates of the tl and br points
    # of this frame cropped from the origin image. x_scale and y_scale
    # are the resized scale.

    bbox[:, 0:2] = bbox[:, 0:2] - im_info[1:3]
    bbox[:, 2:4] = bbox[:, 2:4] * im_info[5:7]

    # change to [x1, y1, x2, y2] and clip
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    w = bbox[:, 2]
    h = bbox[:, 3]
    x2 = x1 + w
    y2 = y1 + h
    refined_box = bbox.copy()
    refined_box[:, 0] = x1.clip(0, im_width - 1)
    refined_box[:, 1] = y1.clip(0, im_height - 1)
    refined_box[:, 2] = x2.clip(0, im_width - 1)
    refined_box[:, 3] = y2.clip(0, im_height - 1)

    # check the boxes whether all in the frame (boxes maybe out of the image view)
    if refined_box.shape[0] > 0:
        if min(refined_box[:, 0]) < 0 or min(refined_box[:, 1]) < 0 or \
                min(refined_box[:, 2]) >= im_width or min(refined_box[:, 3]) >= im_height:
            raise RuntimeError('Box out the image view')

        if ((refined_box[:, 0] <= refined_box[:, 2]).all() == False) or (
                (refined_box[:, 1] <= refined_box[:, 3]).all() == False):
            raise RuntimeError('Find invalid boxes: x1 >= x2 or y1 >= y2')

    return refined_box, gt_class




