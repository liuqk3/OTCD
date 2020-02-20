
from lib.utils.misc import set_visible_devices
GPU_IDS = [0]
set_visible_devices(GPU_IDS)

import argparse
from lib.tracking.tracker import Tracker
import os
import numpy as np
from lib.model.detection_net.rfcn import RFCN
from lib.model.detection_sbc_joint.det_sbc import DET_SBC
from lib.model.tracking_net.rfcn_tracking_single_branch import RFCN_tracking
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list
from lib.model.sbc_net.spatial_binary_classifier import SBC
import torch
import torch.nn as nn
import pprint

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--mot_dir', default='/home/liuqk/Dataset/MOT',
                        help='training dataset', type=str)
    parser.add_argument('--cfg_file',
                        help='optional config file',
                        default='../cfgs/resnet101_ls.yml', type=str)
    parser.add_argument('--set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--vis', default=False,
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--detection_interval', default=1,
                        help='the key frame scheduler', type=int)
    parser.add_argument('--iou_or_appearance', default='both', choices=['iou', 'appearance', 'both'],
                        help='the cost used for tracking', type=str)
    parser.add_argument('--dataset_year', default='MOT16',
                        choices=['MOT16', 'MOT17'],
                        help='the dataset to tracking', type=str)
    parser.add_argument('--detectors', default='PRIVATE',
                        choices=['PRIVATE', 'DPM', 'SDP', 'FRCNN', 'POI'],
                        help='the detections provided by the detectors will be used to tracking', type=str)
    parser.add_argument('--stage', default='val',
                        choices=['val', 'test', 'train'],
                        help='the phase for this running', type=str)

    parser.add_argument('--mv_for_box', default=False, type=bool, action='store_true',
                        help='crop motion vectors for box, if True, it will be used to estimate, the shift of box between adjacent frames')
    parser.add_argument('--im_for_box', default=False, type=bool,
                        action='store_true', help='crop image patch vectors for box')
    parser.add_argument('--res_for_box', default=False, type=bool,
                        action='store_true', help='crop residuals for box')
    parser.add_argument('--vis', default=False, type=bool,
                        action='store_true', help='visualize the tracking results')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # if not (len(GPU_IDS) > 0 and torch.cuda.available()):
    if len(GPU_IDS) == 0:
        args.cuda = False
    else:
        args.cuda = True

    if args.cuda and len(GPU_IDS) > 1:
        args.mGPUs = True
    else:
        args.mGPUs = False

    args.dataset_year = args.dataset_year.split(',') # ['MOT16']  # ['MOT16', 'MOT17']
    args.detectors = args.detectors.split(',') # ['DPM']#, 'SDP', 'FRCNN'] # ['PRIVATE', 'DPM', 'SDP', 'FRCNN', 'POI']
    args.stage = args.stage.split(',') # ['val'] # ['test', 'val']

    print(args.stage)

    # TODO: the following operations will take some time, set them to False if a faster tracker is required.
    args.save_detections_with_feature = False  # save the detections, along with the cropped features

    args.feature_crop_size = (1024, 7, 7)  # appearance crop size (h, w)
    args.mv_crop_size = (2, 120, 40) # mv crop size. (c, h, w)
    args.im_crop_size = (3, 120, 40) # im crop size, (c, h, w)
    args.residual_crop_size = (3, 120, 40)  # residual crop size, (c, h, w)

    args.detection_sbc_model = './save/detection_sbc_101_4_1_9417.pth'

    args.tracking_box_transform_sigma = 1.5
    args.tracking_model = './save/tracking_net_single_resnet18_mv_residual_2_10_6034.pth'

    # ----------------set the configures for the model we trained on motchallenge ------------------------------
    if args.large_scale:
        args.cfg_file = "./cfgs/resnet101_ls.yml"
    else:
        args.cfg_file = "./cfgs/resnet101.yml"
    args.set_cfgs = ['ANCHOR_SCALES', '[1, 4, 8, 16, 32]', 'ANCHOR_RATIOS', '[1, 3]', 'MAX_NUM_GT_BOXES', '55']
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.USE_GPU_NMS = args.cuda
    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    #  ----------------------------- detection sbc model  ----------------------
    detection_sbc_checkpoint = torch.load(args.detection_sbc_model)
    detection_sbc_model = DET_SBC(det_num_layers=101, sbc_crop_size=args.feature_crop_size, pretrained=False)
    detection_sbc_model.load_state_dict(detection_sbc_checkpoint['model'])

    # ----------------------------- detection model --------------------------------
    detection_model = RFCN(classes=np.asarray(['__background__', 'person']),
                           base_net='resnet',
                           num_layers_i=101,
                           pretrained=False,
                           class_agnostic=args.class_agnostic)
    detection_model.create_architecture()
    detection_model.load_state_dict(detection_sbc_model.detection_model.state_dict())
    detection_model.set_train_and_test_configure(phase='test')

    # ---------------------------- tracking model --------------------------------------------
    tracking_model = RFCN_tracking(base_net='resnet', num_layers=18, data_type='mv_residual',
                                   pretrained=False, transform_sigma=args.tracking_box_transform_sigma)
    tracking_model.create_architecture()
    tracking_model.set_train_and_test_configure(phase='test')
    checkpoint = torch.load(args.tracking_model)
    tracking_model.load_state_dict(checkpoint['model'], strict=True)

    # ------------------------- sbc model -------------------------------------
    appearance_model = SBC(input_c=args.feature_crop_size[0],
                           input_h=args.feature_crop_size[1],
                           input_w=args.feature_crop_size[2])
    appearance_model.eval()
    appearance_model.load_state_dict(detection_sbc_model.sbc_model.state_dict())

    if args.cuda:
        detection_model.cuda()
        tracking_model.cuda()
        appearance_model.cuda()
    if args.mGPUs:
        detection_model = nn.DataParallel(detection_model)
        tracking_model = nn.DataParallel(tracking_model)
        appearance_model = nn.DataParallel(appearance_model)

    print('Called with args:')
    print(args)

    tracker = Tracker(base_net_model=detection_model,
                      tracking_model=tracking_model,
                      appearance_model=appearance_model,
                      args=args, cfg=cfg)

    mot_seqs = {
        'MOT16': {
            'train': ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13'],
            'test': ['MOT16-08', 'MOT16-06', 'MOT16-03', 'MOT16-01', 'MOT16-14', 'MOT16-07', 'MOT16-12'],
            'val': ['MOT16-09', 'MOT16-10']
        },

        'MOT17': {
            'train': ['MOT17-05', 'MOT17-04', 'MOT17-02', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13'],
            'test': ['MOT17-12', 'MOT17-01', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-14', 'MOT17-03'],
            'val': ['MOT17-09', 'MOT17-10']
        },

        '2DMOT2015': {
            'train': ['ETH-Sunnyday', 'ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'KITTI-13',
                      'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2'],
            'test': ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher',
                     'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1'],
            'val': []
        }
    }

    mot_seqs_train = mot_seqs['MOT16']['train'] + mot_seqs['2DMOT2015']['train'] + mot_seqs['MOT17']['train']
    mot_seqs_test = mot_seqs['MOT16']['test'] + mot_seqs['2DMOT2015']['test'] + mot_seqs['MOT17']['test']

    mot_info = {
        'MOT16': {'detectors': ['PRIVATE', 'DPM', 'POI'], 'split': ['test', 'train', 'val']},
        'MOT17': {'detectors': ['SDP', 'FRCNN', 'DPM'], 'split': ['test', 'train', 'val']},
        '2DMOT2015': {'detectors': ['DPM'], 'split': ['test', 'train']}
    }

    dataset_year = args.dataset_year if args.dataset_year is not None else ['MOT16', 'MOT17']
    detectors = args.detectors if args.detectors is not None else ['PRIVATE', 'DPM', 'SDP', 'FRCNN', 'POI']
    stage = args.stage if args.stage is not None else ['test', 'val']

    for one_dataset in dataset_year:
        for det_name in detectors:

            if det_name not in mot_info[one_dataset]['detectors']:
                print('dataset {} does not has detector {}'.format(one_dataset, det_name))
                continue

            for s in stage:
                if s not in mot_info[one_dataset]['split']:
                    print('dataset {} does not has stage {}'.format(one_dataset, stage))
                    continue

                subset = mot_seqs[one_dataset][s]
                for seq in subset:
                    if seq in mot_seqs_train:
                        s = 'train'
                    elif seq in mot_seqs_test:
                        s = 'test'

                    if one_dataset == 'MOT17':
                        seq = seq + '-' + det_name

                    print('tracking on ' + seq + ' using ' + det_name + ' detector ...')

                    video_file = os.path.join(args.mot_dir, one_dataset, s, seq, seq + '.mp4')
                    if not os.path.exists(video_file):
                        raise RuntimeError(video_file + ' does not exists')

                    if one_dataset == 'MOT17':
                        tracking_output_path = os.path.join('./track_results/',
                                                            str(args.tracking_box_transform_sigma),
                                                            str(args.detection_interval),
                                                            args.iou_or_appearance, one_dataset, s)
                    else:
                        tracking_output_path = os.path.join('./track_results/',
                                                            str(args.tracking_box_transform_sigma),
                                                            str(args.detection_interval),
                                                            args.iou_or_appearance, one_dataset, det_name, s)

                    detection_output_path = os.path.join('./detect_results',
                                                         str(args.detection_interval),
                                                         args.iou_or_appearance, one_dataset, det_name, s)
                    if not os.path.exists(tracking_output_path):
                        os.makedirs(tracking_output_path)
                    if not os.path.exists(detection_output_path):
                        os.makedirs(detection_output_path)

                    time_file = os.path.join(tracking_output_path, 'time_analysis' + det_name + '.txt')
                    tracking_output_file = os.path.join(tracking_output_path, seq + '.txt')
                    detection_output_file = os.path.join(detection_output_path, seq + '.txt')

                    tracker.track_on_video(video_file=video_file,
                                           tracking_output_file=tracking_output_file,
                                           detection_output_file=detection_output_file,
                                           detector_name=det_name)

                tracker.save_time(time_file)
