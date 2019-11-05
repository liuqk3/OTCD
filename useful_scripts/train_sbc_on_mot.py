from lib.utils.misc import set_visible_devices
GPU_IDS = [7]
set_visible_devices(GPU_IDS)


import argparse
import torch
import torch.nn as nn
# from lib.model.sbc_net.st_rnn import ST_RNN
import os
import numpy as np
from lib.datasets.motchallenge_tracklets import motchallenge_tracklets
from lib.model.utils.config import cfg
from torch.autograd import Variable
from lib.model.utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient
import random
import pdb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ST_RNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='motchallenge', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    args.dataset = 'motchallenge'
    args.save_dir = '../save'
    args.arch = 'sbc_v6'
    args.cuda = True if len(GPU_IDS) > 0 else False
    args.mGPUs = True if len(GPU_IDS) > 1 else False

    # define some attributes for the dataset generator
    args.smooth_loss = True#False
    args.feature_type = 'origin' # the origin feature or warped feature
    args.crop_w = 7
    args.crop_h = 7
    args.crop_c = 1024 # the number of channels of feature
    args.num_tracklets = 1e4
    args.negative_positive_ratio = 1
    args.max_length = 2
    args.min_length = 2
    args.interval = 1 # only used for 'tracklets', not 'pair'
    args.jitter = True
    args.iou_thr = 0.7 # the iou threshold jitter the boxes
    args.shuffle = False # whether to shuffle the tracklets, only useful 'tracklet', not 'pair'
    args.regenerate = 2 # how many epoch to regenerate the train data
    args.tracklet_or_pair = 'tracklet' # 'pair' # 'tracklet'

    args.session = 7
    args.optimizer = 'sgd'
    args.clip_grad = 5
    args.batch_size = 2
    args.lr = 1e-4
    args.lr_decay_step = 10
    args.lr_decay_gamma = 0.1

    args.disp_interval = 5

    args.start_epoch = 1
    args.max_epochs = 20

    # resume a training
    args.resume = False
    args.new_resume = True  # if true, just load the weight of the network, not the lr, session, or itr
    args.checksession = 4
    args.checkepoch = 15
    args.checkpoint = 4999

    # ------------------------ generate the dataset --------------------------------
    mot_tracklets = motchallenge_tracklets(phase='train', name='motchallenge_tracklet_sbc')
    mot_tracklets.generate_tracklets(num_tracklets=args.num_tracklets,
                                     negative_positive_ratio=args.negative_positive_ratio,
                                     max_length=args.max_length, min_length=args.min_length,
                                     interval=args.interval, jitter=args.jitter, iou_thr=args.iou_thr,
                                     shuffle=args.shuffle, regenerate=False, tracklet_or_pair=args.tracklet_or_pair)

    # ------------------------ define the model ------------------------------------
    if args.arch == 'sbc_v6':
        from lib.model.sbc_net.spatial_binary_classifier import SBC
        sbc_model = SBC(input_h=args.crop_h, input_w=args.crop_w, input_c=args.crop_c)
    sbc_model.train()

    # ----------------------- define the optimizer ---------------------------------
    lr = args.lr
    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(sbc_model.parameters(), lr=lr)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(sbc_model.parameters(), lr=lr, momentum=cfg.TRAIN.MOMENTUM)

    # ----------------------------- resume the training ------------------------------
    output_dir = os.path.join(args.save_dir, args.arch, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.resume:
        load_name = '{}_{}_{}_{}.pkl'.format(args.arch, args.checksession, args.checkepoch, args.checkpoint)
        load_path = os.path.join(output_dir, load_name)
        if not os.path.exists(load_path):
            raise RuntimeError('Pretrained model {} does not exist!'.format(load_path))

        checkpoint = torch.load(load_path)

        sbc_model.load_state_dict(checkpoint['model'])
        # if not start a new session
        if not args.new_resume:
            # args.session = checkpoint['session']
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = np.unique(np.array([optimizer.param_groups[idx]['lr'] for idx in range(len(optimizer.param_groups))]))
            #lr = optimizer.param_groups[0]['lr']
            print("loaded checkpoint %s" % (load_name))

    if args.cuda:
        sbc_model.cuda()
    if args.mGPUs:
        sbc_model = nn.DataParallel(sbc_model)

    # ---------------------------- make the inputs variable here ----------------
    tracklet_feature = torch.FloatTensor([0])
    tracklet_label = torch.FloatTensor([0])

    if args.cuda:
        tracklet_feature = tracklet_feature.cuda()
        tracklet_label = tracklet_label.cuda()

    tracklet_feature = Variable(tracklet_feature)
    tracklet_label = Variable(tracklet_label)

    # ---------------------------- train the model-------------------------------
    num_itr = len(mot_tracklets.tracklets) // args.batch_size

    for epoch in range(args.start_epoch, args.max_epochs + 1):

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        if epoch % (args.regenerate + 1) == 0:
            mot_tracklets.generate_tracklets(num_tracklets=args.num_tracklets,
                                             negative_positive_ratio=args.negative_positive_ratio,
                                             max_length=args.max_length, min_length=args.min_length,
                                             interval=args.interval, jitter=args.jitter, iou_thr=args.iou_thr,
                                             shuffle=args.shuffle, regenerate=True,
                                             tracklet_or_pair=args.tracklet_or_pair)

        # shuffle the tracklets
        random.shuffle(mot_tracklets.tracklets)

        for itr in range(num_itr):
            # Train one batch. Because the length of tracklet may be different,
            # hence we implement the batch by for loop
            loss = Variable(tracklet_feature.data.new([0]), requires_grad=True)
            for b in range(args.batch_size):
                idx = int(itr * args.batch_size) + b
                input_data = mot_tracklets.get_rnn_input(idx, args.feature_type) # tuple

                tracklet_feature.data.resize_(input_data[0].size()).copy_(input_data[0]).contiguous()
                tracklet_label.data.resize_(input_data[1].size()).copy_(input_data[1]).contiguous()

                # take the first feature in this tracklet as the initial hidden state
                feature_1 = tracklet_feature[0]
                feature_2 = tracklet_feature[1]
                cls_score, cls_prob = sbc_model(feature_1, feature_2)
                loss = loss + sbc_model.get_loss(cls_score, tracklet_label.long(), smooth=args.smooth_loss)
                # cls_score = cls_score.detach()
            loss = loss / args.batch_size

            # update the parameter
            optimizer.zero_grad()
            loss.backward()

            if args.clip_grad is not None:
                #nn.utils.clip_grad_norm(st_rnn.parameters(), args.clip_grad)
                clip_gradient(sbc_model, args.clip_grad)
            optimizer.step()


            if sbc_model.conv_cls.weight.grad.data is not None:
                conv_cls_grad = sbc_model.conv_cls.weight.grad.data
            else:
                conv_cls_grad = sbc_model.conv_cls.weight.data.new([0])
            if (conv_cls_grad != conv_cls_grad).sum() > 0:
                raise RuntimeError('\n there is nan in the grad of one layer\n')
                # print('\n there is nan in the grad of one layer\n')
                # pdb.set_trace()

            if cls_prob.max().data[0] == 1:
                print('Find probability of 1, maybe there are some thing wrong!')
                #pdb.set_trace()

            if (itr + 1) % args.disp_interval == 0:
                print('---------[session %d] [epoch %2d] [iter %4d/%4d]----------------' % (args.session, epoch, itr + 1, num_itr))
                print('loss: %.5f' % (loss.data[0]))
                if args.cuda:
                    prob = cls_prob[:, tracklet_label.long()].cpu()
                    # label = tracklet_label.cpu()
                else:
                    prob = cls_prob[:, tracklet_label.long()]
                    # label = tracklet_label
                print('class prob:', prob.data.numpy())
                print('lr: {}'.format(lr))
                # print('label:', label.data.numpy())

        # save the model
        "kuai gan huohjghghghghghghghjkjllljljljljkljljlj,wo shi yangwenfei"
        save_name = '{}_{}_{}_{}.pkl'.format(args.arch, args.session, epoch, itr)
        save_name = os.path.join(output_dir, save_name)
        if args.mGPUs:
            trained_weight = sbc_model.module.state_dict()
        else:
            trained_weight = sbc_model.state_dict()
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': trained_weight,
            'optimizer': optimizer.state_dict()
        }, save_name)










