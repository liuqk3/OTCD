# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.utils.misc import *
import math

GPU_IDS = [9]
set_visible_devices(GPU_IDS)

import argparse
import os
import pdb
import pprint
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list
from lib.model.utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient
from lib.roi_data_layer.roibatchLoader_pair import roibatchLoader as roibatchLoader_pair
from lib.roi_data_layer.roibatchLoader_tracking_pair import roibatchLoader as roibatchLoader_tracking_pair
from lib.roi_data_layer.roibatchLoader_single import roibatchLoader as roibatchLoader_single
from lib.roi_data_layer.roidb_single import combined_roidb as combined_roidb_single
from lib.roi_data_layer.roidb_pair import combined_roidb as combined_roidb_pair


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--arch', dest='arch', default='rcnn', choices=['rcnn', 'rfcn', 'couplenet'])
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=5, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="save",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--cuda', dest='cuda', default=True,
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--ohem', dest='ohem',
                        help='Use online hard example mining for training',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=2, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--cfg_file', dest='cfg_file',
                        default=None, help='the file to config the detection net')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':

    args = parse_args()

    args.dataset = 'motchallenge'
    args.arch = 'tracking_net_single'
    args.transform_sigma = 1.5
    args.save_dir = '../save'
    args.net = 'resnet'
    args.num_layers = 18
    args.data_type = 'residual'
    args.cuda = True if len(GPU_IDS) > 0 else False
    args.mGPUs = True if len(GPU_IDS) > 1 else False
    args.session = 5
    args.batch_size = 4
    args.lr = 1e-4
    args.lr_decay_step = 6
    args.clip_grad = 10

    args.disp_interval = 5

    args.start_epoch = 1
    args.max_epochs = 20

    # resume a training
    args.resume = False
    args.new_resume = False  # if true, just load the weight of the network, not the lr, session, or itr
    args.resume_num_layers = 18
    args.checksession = 1
    args.checkepoch = 18
    args.checkpoint = 6034

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.use_tfboard:
        from lib.model.utils.logger import Logger

        # Set the logger
        logger = Logger('./logs')

    if args.dataset in ['mot', 'motchallenge']:
        name1 = 'motchallenge_tracking_pair_train'
        args.imdb_name = name1
        # in fact, we found that the max_num_gt_boxes in MOT2015 and MOT17 is 52
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '55', 'RESNET.FIXED_BLOCKS', 0]

    # ----------------------- define configuration -------------------------------------
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if args.cuda:
        cfg.CUDA = True
    # Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # -----------------------------define image database --------------------------------
    if 'pair' in args.imdb_name:
        imdb, roidb, ratio_list, ratio_index = combined_roidb_pair(args.imdb_name)
        train_size = len(roidb)
        print('{:d} roidb entries'.format(len(roidb)))
        sampler_batch = sampler(train_size, args.batch_size)
        dataset = roibatchLoader_tracking_pair(roidb=roidb, ratio_list=ratio_list, ratio_index=ratio_index,
                                      batch_size=args.batch_size, num_classes=imdb.num_classes,
                                      training=True, im_type=['mv', 'residual'])
    else:
        raise RuntimeError('unrecognized imdb name: {}'.format(args.imdb_name))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)

    # -------------------------- initilize the network here -------------------------------
    if args.arch == 'tracking_net_single' and args.net == 'resnet':
        from lib.model.tracking_net.rfcn_tracking_single_branch import RFCN_tracking
        if args.transform_sigma == 0.707:
            sigma = 1 / math.sqrt(2)
        else:
            sigma = args.transform_sigma
        cnn_model = RFCN_tracking(base_net=args.net,
                                  num_layers=args.num_layers,
                                  data_type=args.data_type,
                                  pretrained=True,
                                  transform_sigma=sigma)
    else:
        print("network is not defined")

    cnn_model.create_architecture()
    cnn_model.set_train_and_test_configure(phase='train')

    # ------------- define the optimizer ----------------
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(cnn_model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # ---------------- resume the training --------------------------
    output_dir = os.path.join(args.save_dir, args.arch, args.net, args.dataset, str(args.transform_sigma))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.resume:
        load_name = os.path.join(output_dir, '{}_{}{}_{}_{}_{}_{}.pth'.format(args.arch,
                                                                              args.net,
                                                                              args.resume_num_layers,
                                                                              args.data_type,
                                                                              args.checksession,
                                                                              args.checkepoch,
                                                                              args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        cnn_model.load_state_dict(checkpoint['model'])

        if not args.new_resume:
            args.session = checkpoint['session']
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = np.array([optimizer.param_groups[idx]['lr'] for idx in range(len(optimizer.param_groups))])
            #lr = optimizer.param_groups[0]['lr']
            if 'pooling_mode' in checkpoint.keys():
                cfg.POOLING_MODE = checkpoint['pooling_mode']
            print("loaded checkpoint %s" % (load_name))
    if args.cuda:
        cnn_model.cuda()
    if args.mGPUs:
        cnn_model = nn.DataParallel(cnn_model)

    # ----------------------- initialize the tensor holder here -----------------------------
    #  Dataloader return back the following data
    im_info = torch.FloatTensor(1)
    frame_1 = torch.FloatTensor(1)
    frame_1_box = torch.FloatTensor(1)
    num_box_1 = torch.FloatTensor(1)
    frame_2 = torch.FloatTensor(1)
    frame_2_box = torch.FloatTensor(1)
    num_box_2 = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_info = im_info.cuda()
        im_info = im_info.cuda()

        frame_1 = frame_1.cuda()
        frame_1_box = frame_1_box.cuda()
        num_box_1 = num_box_1.cuda()
        frame_2 = frame_2.cuda()
        frame_2_box = frame_2_box.cuda()
        num_box_2 = num_box_2.cuda()

    # make variable
    im_info = Variable(im_info)
    frame_1 = Variable(frame_1)
    frame_1_box = Variable(frame_1_box)
    num_box_1 = Variable(num_box_1)
    frame_2 = Variable(frame_2)
    frame_2_box = Variable(frame_2_box)
    num_box_2 = Variable(num_box_2)

    # ------------------------- begin to train -----------------------------
    iters_per_epoch = int(train_size / args.batch_size)
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        dataset.resize_batch()
        # setting to train mode
        cnn_model.train()
        loss_temp = 0
        start = time.time()

        if epoch >= 7:
            args.base_feat_loss_weight = 0

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):

            data = next(data_iter)

            # im_info, frame_1, frame_1_box, num_box_1, frame_2, frame_2_box, num_box_2

            # # -------------------------------------------------------------------------
            # # show the data to see whether it has processed correctly
            # import matplotlib.pyplot as plt
            # from lib.utils.misc import warp_from_offsets
            # from lib.model.utils.net_utils import vis_detections
            # mean_pixel = np.array([[[102.9801, 115.9465, 122.7717]]])
            #
            # i_im = data[1][:, 0:3, :, :] # [bs, 3, h, w]
            # # mv = data[4][:, 3:5, :, :] # [bs, 2, h, w]
            # # residual = data[4][:, 5:8, :, :]
            #
            # # warped = warp_from_offsets(mv, i_im)
            #
            # # warped_r = warped + residual # [bs, 3, h, w]
            #
            # # plt.figure()
            # idx = 0
            # one_i_im = np.array(i_im[idx].permute(1, 2, 0).numpy() + mean_pixel, dtype=np.uint8)
            # one_i_box = data[2][idx].numpy()
            # one_i_im = vis_detections(one_i_im.copy(), 'person', one_i_box, thresh=-2)
            # plt.imshow(one_i_im)
            # plt.title('i frame')
            # plt.axis('off')
            # plt.show()
            # # plt.savefig('/home/liuqk/Program/pycharm/R-FCN-pytorch_v5/images/i_frame.pdf',)
            #
            # # p_im = data[4][:, 0:3, :, :] # [bs, 3, h, w]
            # # one_p_im = np.array(p_im[idx].permute(1, 2, 0).numpy() + mean_pixel, dtype=np.uint8)
            # # one_p_box = data[5][idx].numpy()
            # # one_p_im = vis_detections(one_p_im.copy(), 'person', one_p_box, thresh=-2)
            # # plt.imshow(one_p_im)
            # # plt.title('p frame')
            # # plt.axis('off')
            # # plt.savefig('/home/liuqk/Program/pycharm/R-FCN-pytorch_v2/images/p_frame.pdf', )
            # #
            # # one_warped_r = np.array(warped_r[idx].permute(1, 2, 0).numpy() + mean_pixel, dtype=np.uint8)
            # # one_warped_r = vis_detections(one_warped_r.copy(), 'person', one_p_box, thresh=-2)
            # # plt.imshow(one_warped_r)
            # # plt.title('p frame (warped)')
            # # plt.axis('off')
            # # plt.savefig('/home/liuqk/Program/pycharm/R-FCN-pytorch_v2/images/i_frames_warp.pdf', dpi=100)
            # # # ------------------------- end of show -----------------------------------------
            #
            #
            #
            # # # -------------------------------------------------------------
            # # # show the data to see whether it has processed correctly
            # # import matplotlib.pyplot as plt
            # # from lib.utils.misc import warp_from_offsets
            # # from lib.model.utils.net_utils import vis_detections
            # # mean_pixel = np.array([[[102.9801, 115.9465, 122.7717]]])
            # #
            # # i_im = data[1][:, 0:3, :, :] # [bs, 3, h, w]
            # # mv = data[4][:, 3:5, :, :] # [bs, 2, h, w]
            # # residual = data[4][:, 5:8, :, :]
            # #
            # # warped = warp_from_offsets(mv, i_im)
            # #
            # # warped_r = warped + residual # [bs, 3, h, w]
            # #
            # # plt.figure()
            # # idx = 0
            # # one_i_im = np.array(i_im[idx].permute(1, 2, 0).numpy() + mean_pixel, dtype=np.uint8)
            # # one_i_box = data[2][idx].numpy()
            # # one_i_im = vis_detections(one_i_im.copy(), 'person', one_i_box, thresh=-2)
            # # plt.imshow(one_i_im)
            # # plt.title('i frame')
            # # plt.axis('off')
            # # plt.savefig('/home/liuqk/Program/pycharm/R-FCN-pytorch_v2/images/i_frame.pdf',)
            # #
            # # p_im = data[4][:, 0:3, :, :] # [bs, 3, h, w]
            # # one_p_im = np.array(p_im[idx].permute(1, 2, 0).numpy() + mean_pixel, dtype=np.uint8)
            # # one_p_box = data[5][idx].numpy()
            # # one_p_im = vis_detections(one_p_im.copy(), 'person', one_p_box, thresh=-2)
            # # plt.imshow(one_p_im)
            # # plt.title('p frame')
            # # plt.axis('off')
            # # plt.savefig('/home/liuqk/Program/pycharm/R-FCN-pytorch_v2/images/p_frame.pdf', )
            # #
            # # one_warped_r = np.array(warped_r[idx].permute(1, 2, 0).numpy() + mean_pixel, dtype=np.uint8)
            # # one_warped_r = vis_detections(one_warped_r.copy(), 'person', one_p_box, thresh=-2)
            # # plt.imshow(one_warped_r)
            # # plt.title('p frame (warped)')
            # # plt.axis('off')
            # # plt.savefig('/home/liuqk/Program/pycharm/R-FCN-pytorch_v2/images/i_frames_warp.pdf', dpi=100)
            # # # ------------------------- end of show -----------------------------------------

            im_info.data.resize_(data[0].size()).copy_(data[0]).contiguous()
            frame_1.data.resize_(data[1].size()).copy_(data[1]).contiguous()
            frame_1_box.data.resize_(data[2].size()).copy_(data[2]).contiguous()
            num_box_1.data.resize_(data[3].size()).copy_(data[3]).contiguous()
            frame_2_box.data.resize_(data[5].size()).copy_(data[5]).contiguous()
            num_box_2.data.resize_(data[6].size()).copy_(data[6]).contiguous()

            if args.data_type == 'mv_residual':
                frame_2.data.resize_(data[4].size()).copy_(data[4]).contiguous()
            elif args.data_type == 'mv':
                mv = data[4][:, 0:2, :, :]
                frame_2.data.resize_(mv.size()).copy_(mv).contiguous()
            elif args.data_type == 'residual':
                residual = data[4][:, 2:5, :, :]
                frame_2.data.resize_(residual.size()).copy_(residual).contiguous()


            # noted the num_box_1 are the same with num_box_2
            if (frame_1_box[:, :, 5] - frame_2_box[:, :, 5]).data.abs().sum() != 0 or\
                    (num_box_1 - num_box_2).data.abs().sum() != 0:
                raise RuntimeError('Invalide combination of frames!')

            outputs = cnn_model(frame_1_box, frame_2, frame_2_box, num_box_1) # (bbox_pred, loss_bbox)

            cnn_model.zero_grad()

            bbox_pred, loss_bbox = outputs

            loss = loss_bbox.mean()

            loss_temp += loss.data[0]

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad is not None:
                clip_gradient(cnn_model, args.clip_grad)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    bbox_loss = loss_bbox.mean().data[0]
                else:
                    bbox_loss = loss_bbox.data[0]

                print("---------------- [session %d][epoch %2d][iter %4d/%4d] ------------------" % (args.session, epoch, step, iters_per_epoch))
                print("loss: %.4f, lr: {}, num_boxes: %d, time cost: %f".format(change_to_string(np.unique(lr))) % (loss_temp, num_box_1.data.sum(), end - start))
                if args.use_tfboard:
                    info = {
                        'loss_bbox': loss_bbox
                    }
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, step)

                loss_temp = 0
                start = time.time()

        if args.mGPUs:
            save_name = os.path.join(output_dir,
                                     '{}_{}{}_{}_{}_{}_{}.pth'.format(args.arch,
                                                                      args.net,
                                                                      args.num_layers,
                                                                      args.data_type,
                                                                      args.session,
                                                                      epoch,
                                                                      step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': cnn_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        else:
            save_name = os.path.join(output_dir,
                                     '{}_{}{}_{}_{}_{}_{}.pth'.format(args.arch,
                                                                      args.net,
                                                                      args.num_layers,
                                                                      args.data_type,
                                                                      args.session,
                                                                      epoch,
                                                                      step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': cnn_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        print('save model: {}'.format(save_name))

        end = time.time()
        print(end - start)
