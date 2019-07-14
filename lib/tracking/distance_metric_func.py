# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment
from .utils import *
import math


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : 1D array
        A bounding box in format `[x1, y1 ,x2, y2]`.
    candidates : 2D tensor
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        1D tensor, the intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox = bbox.unsqueeze(dim=0)
    bbox = bbox.repeat(candidates.size()[0], 1)

    bbox_tl, bbox_br = bbox[:, 0:2], bbox[:, 2:4]
    candidates_tl, candidates_br = candidates[:, 0:2], candidates[:, 2:4]

    tl = torch.stack((bbox_tl, candidates_tl), dim=0)
    tl, _ = tl.max(dim=0)

    br = torch.stack((bbox_br, candidates_br), dim=0)
    br, _ = br.min(dim=0)

    wh = br - tl
    wh[wh < 0] = 0

    area_intersection = torch.prod(wh, dim=1)
    area_bbox = torch.prod(bbox[:, 2:4] - bbox[:, 0:2] + 1, dim=1)
    area_candidates = torch.prod(candidates[:, 2:4] - candidates[:, 0:2] + 1, dim=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def euclidean_distance(bbox, candidates, image_size):
    """Computer Euclidean distance.

    Parameters
    ----------
    bbox : 1D array
        A bounding box in format `[x1, y1 ,x2, y2]`.
    candidates : 2D tensor
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.
    image_size: 1D tensor, with size [2]
        [h, w] of the image, used to normalize the distance

    Returns
    -------
    ndarray
        1D tensor, the Euclidean distance between the `bbox` and each
        candidate. A higher score means a larger distance between the
        `bbox` and candidate.

    """
    bbox = bbox.unsqueeze(dim=0) # [1, 4]
    bbox = bbox.repeat(candidates.size()[0], 1) # [num_candidate, 4]

    bbox_tl, bbox_br = bbox[:, 0:2], bbox[:, 2:4]
    bbox_ctr = bbox_tl + (bbox_br - bbox_tl + 1) / 2 - 0.5

    candidates_tl, candidates_br = candidates[:, 0:2], candidates[:, 2:4]
    candidates_ctr = candidates_tl + (candidates_br - candidates_tl + 1) / 2 - 0.5

    diff = bbox_ctr - candidates_ctr
    dist = torch.norm(diff, p=2, dim=1)

    # normalize the dist
    diag_line = math.sqrt(image_size[0] * image_size[1])

    dist = dist / diag_line

    return dist


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[track.Track]
        A list of tracks.
    detections : List[detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    2D tensor
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # get the detection candidates
    bbox = tracks[0].to_tlbr()
    candidates = torch.zeros(0, 4).float()
    if bbox.is_cuda:
        candidates = candidates.cuda()
    for i in detection_indices:
        one_candidate = torch.unsqueeze(detections[i].to_tlbr(), dim=0)
        candidates = torch.cat((candidates, one_candidate), 0)

    # get the cost matrix
    cost_matrix = torch.zeros((len(track_indices), len(detection_indices))).float()
    for row, track_idx in enumerate(track_indices):
        bbox = tracks[track_idx].to_tlbr()
        # candidates = torch.zeros(0, 4).float()
        # if bbox.is_cuda:
        #     candidates = candidates.cuda()
        # for i in detection_indices:
        #     one_candidate = torch.unsqueeze(detections[i].to_tlbr(), dim=0)
        #     candidates = torch.cat((candidates, one_candidate), 0)

        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix


def euclidean_distance_cost(tracks, detections, image_size,
                            track_indices=None,
                            detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[track.Track]
        A list of tracks.
    detections : List[detection.Detection]
        A list of detections.
    image_size: 1D tensor, with size [2]
        [h, w] of the image, used to normalize the distance
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    2D tensor
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        the normalized euclidean distance between tracks[track_indices[i]]
        and detections[detection_indices[j]].

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # get the detection candidates
    bbox = tracks[0].to_tlbr()
    candidates = torch.zeros(0, 4).float()
    if bbox.is_cuda:
        candidates = candidates.cuda()
    for i in detection_indices:
        one_candidate = torch.unsqueeze(detections[i].to_tlbr(), dim=0)
        candidates = torch.cat((candidates, one_candidate), 0)

    cost_matrix = torch.zeros((len(track_indices), len(detection_indices))).float()
    for row, track_idx in enumerate(track_indices):
        bbox = tracks[track_idx].to_tlbr()
        # candidates = torch.zeros(0, 4).float()
        # if bbox.is_cuda:
        #     candidates = candidates.cuda()
        # for i in detection_indices:
        #     one_candidate = torch.unsqueeze(detections[i].to_tlbr(), dim=0)
        #     candidates = torch.cat((candidates, one_candidate), 0)

        cost_matrix[row, :] = euclidean_distance(bbox, candidates, image_size)
    return cost_matrix



