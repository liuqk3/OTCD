
"""
In this file, we get a correlation feature for each input feature, and concate two input features as
well as two corr features. But we use the classifier in rfcn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from lib.model.psroi_pooling.modules.psroi_pool import PSRoIPool
from lib.model.correlation.modules.correlation import Correlation
from torch.autograd import Variable
from lib.model.roi_align.roi_align.roi_align import RoIAlign
import math
import time


class SBC(nn.Module):  # spatial binary classifier
    def __init__(self, input_h, input_w, input_c, num_classes=2):
        super(SBC, self).__init__()

        if input_h != input_w:
            raise ValueError('The height and width of the input feature are not he same, Only support a square input!')

        self.input_h = input_h
        self.input_w = input_w
        self.input_c = input_c
        # box = torch.FloatTensor([[0, 0, input_w, input_h]]).repeat(3000, 1).cuda()
        # batch_idx = torch.FloatTensor(range(3000)).unsqueeze(dim=1).cuda()  # [bs, 1]
        # self.box = Variable(torch.cat((batch_idx, box), dim=1))


        self.num_classes = num_classes # binary classification

        # define some modules
        # conv layer to produce feature for classification
        self.conv1 = nn.Conv2d(in_channels=2 * (self.input_c + self.input_w * self.input_h),  # features are concated
                               out_channels=1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1024,  # features are concated
                               out_channels=512, kernel_size=3, padding=1)

        self.conv_cls = nn.Conv2d(in_channels=512,
                                  out_channels=self.num_classes * self.input_h * self.input_w,
                                  kernel_size=1, stride=1, padding=0, bias=False)

        self.psroi_pool_cls = PSRoIPool(pooled_height=self.input_h,
                                        pooled_width=self.input_w,
                                        spatial_scale=1, group_size=self.input_h,
                                        output_dim=self.num_classes)

        self.pooling = nn.AvgPool2d(kernel_size=(self.input_h, self.input_w))

        self.relu = nn.ReLU(inplace=True)

        self._init_weigths()

    def _init_weigths(self):

        #
        # def normal_init(m, mean, stddev, truncated=False):
        #     """
        #     weight initalizer: truncated normal and random normal.
        #     """
        #     # x is a parameter
        #     if truncated:
        #         m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        #     else:
        #         m.weight.data.normal_(mean, stddev)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #
        # normal_init(self.conv_cls, 0, 0.01, False)
        # normal_init(self.fc1, 0, 0.01, False)
        # normal_init(self.fc2, 0, 0.01, False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _get_correlation_feature(self, feature_1, feature_2):

        # bs, channels = feature_1.size()[0], feature_1.size()[1]
        #
        # feature_1 = F.normalize(feature_1, dim=1, p=2)
        # feature_2 = F.normalize(feature_2, dim=1, p=2)
        #
        # feature_1_tmp = feature_1.permute(0, 2, 3, 1) # [bs, h1, w1, c]
        # feature_2_tmp = feature_2.contiguous().view(bs, channels, -1) # [bs, c, h2xw2]
        #
        # corr_f = None
        # for b in range(bs):
        #     corr_f_tmp = torch.matmul(feature_1_tmp[b], feature_2_tmp[b])  # [h1, w1, h2xw2]
        #     corr_f_tmp = corr_f_tmp.unsqueeze(dim=0) # [1, h1, w1, h2xw2]
        #     if b == 0:
        #         corr_f = corr_f_tmp #.clone() # [1, h1, w1, h2xw2]
        #     else:
        #         corr_f = torch.cat((corr_f, corr_f_tmp), dim=0) # [bs, h1, w1, h2xw2]
        # corr_f = corr_f.permute(0, 3, 1, 2) # [bs, h2xw2, h1, w1]
        #
        # return corr_f

        bs, channels = feature_1.size(0), feature_1.size(1)
        #print('sbc batch size: {}'.format(bs))

        feature_1 = F.normalize(feature_1, dim=1, p=2)
        feature_2 = F.normalize(feature_2, dim=1, p=2)
        feature_1_tmp = feature_1.unsqueeze(dim=2) # [bs, c, 1, h1, w1]

        feature_2_tmp = feature_2.contiguous().view(bs, channels, -1) # [bs, c, h2xw2]

        feature_2_tmp = feature_2_tmp.unsqueeze(dim=3).unsqueeze(dim=4) # [bs, c, h2xw2, 1, 1]

        max_bs = 300
        num_bs = bs // max_bs

        if num_bs == 0:
            corr = feature_1_tmp * feature_2_tmp # [bs, c, h2xw2, h1, w1]
            corr = corr.sum(dim=1)
        else:
            for i in range(num_bs + 1):
                start = i * max_bs
                end = min((i+1)*max_bs, bs)

                if start != end:

                    corr_tmp = feature_1_tmp[start:end, :, :, :] * feature_2_tmp[start:end, :, :, :]
                    corr_tmp = corr_tmp.sum(dim=1)
                    if i == 0:
                        corr = corr_tmp.clone()
                    else:
                        corr = torch.cat((corr, corr_tmp), dim=0)  # [bs, h1, w1, h2xw2]

        # corr = feature_1_tmp * feature_2_tmp # [bs, c, h2xw2, h1, w1]
        # corr = corr.sum(dim=1)

        return corr


    def forward(self, feature_1, feature_2):
        """
        :param feature_1: [bs, h, w, c], the input feature of an object
        :param feature_2: [bs, h, w, c], the hidden feature of an object, which can be treated as template of an object
        :return:
        """
        if self.training:
            conv_cls_weight = self.conv_cls.state_dict()['weight']
            if (conv_cls_weight != conv_cls_weight).sum() > 0:
                raise RuntimeError('\n there is nan in the weight of one layer\n')

        feature_1 = F.relu(feature_1)
        feature_2 = F.relu(feature_2)

        corr_feature_1 = self._get_correlation_feature(feature_1, feature_2)
        corr_feature_2 = self._get_correlation_feature(feature_2, feature_1)

        # concate features to perform binary classifier
        feature = torch.cat((feature_1, corr_feature_1, feature_2, corr_feature_2), dim=1)  # [bs, 2c+2w*h, h, w]

        feature = self.conv1(feature)
        feature = self.relu(feature)
        feature = self.conv2(feature)
        feature = self.relu(feature)
        feature = self.conv_cls(feature)  # [bs, 512, h, w]
        feature = self.relu(feature)

        bs, channels, height, width = feature.size()[0], feature.size()[1], feature.size()[2], feature.size()[3]

        box = feature.data.new([[0, 0, width, height]]).repeat(bs, 1)
        batch_idx = box.new(range(bs)).unsqueeze(dim=1)  # [bs, 1]
        box = Variable(torch.cat((batch_idx, box), dim=1))

        cls_base_f = self.psroi_pool_cls(feature, box) # [bs, 2, self.input_h, self.input_w]
        cls_score = self.pooling(cls_base_f) # [bs, 2, 1, 1]
        cls_score = cls_score.view(feature.size(0), self.num_classes) # [bs, 2]

        cls_prob = F.softmax(cls_score, dim=1)  # [bs, 2], (prob_false, prob_true)

        if self.training:
            return cls_score, cls_prob
        else:
            return cls_prob, corr_feature_1, corr_feature_2

    def get_loss(self, cls_score, label, smooth=False, epsilon=0.1):
        """
        This function get the loss for the task of classification. We use the binary cross-entropy loss.
        :param cls_score: array_like Variable, [bs, num_class], the score before softmax
        :param label: [bs], the groundtruth label of the samples
        :param smooth: bool, whether to use the smoothed binary cross entropy
        :return: Variable, the loss
        """

        # num_samples, num_class = cls_score.size()[0], cls_score.size()[1]
        # loss = - F.log_softmax(cls_score, dim=1)
        #
        # label_expand = label.unsqueeze(dim=1).repeat(1, num_class)
        #
        # class_id = torch.LongTensor(range(num_class)).unsqueeze(dim=0)
        # class_id = class_id.repeat(num_samples, 1)
        #
        # if label.data.is_cuda:
        #     class_id = class_id.cuda()
        # class_id = Variable(class_id, requires_grad=False)
        #
        # mask = class_id == label_expand
        # loss = loss[mask].mean()

        # loss = F.nll_loss(F.log_softmax(cls_score, 1), label)
        if not smooth:
            loss = F.cross_entropy(cls_score, label)
        elif smooth:
            if isinstance(epsilon, Variable):
                epsilon = epsilon.data[0]

            # get the batch size and number of classes
            batch_size, num_classes = cls_score.size()[0], cls_score.size()[1]

            # create the smooth label
            smooth_label = cls_score.data.new(batch_size, num_classes).fill_(0)

            label_expand = label.unsqueeze(dim=1).repeat(1, num_classes).data # [bs, num_classes]
            class_id = label.data.new(range(num_classes)).unsqueeze(dim=0) # [1, num_classes]
            class_id = class_id.repeat(batch_size, 1) # [bs, num_classes]
            # class_id = Variable(class_id, requires_grad=False) # [bs, num_classes]

            smooth_label[class_id != label_expand] = (1.0 / num_classes) * epsilon
            smooth_label[class_id == label_expand] = 1.0 - ((num_classes - 1) / num_classes * epsilon)
            smooth_label = Variable(smooth_label, requires_grad=False)  # shape: (batch_size, C)

            # get the probability
            loss = - F.log_softmax(cls_score, dim=1)  # softmax the input

            loss = torch.matmul(loss, smooth_label.permute(1, 0))  # shape (batch_size, batch_size)
            loss = torch.diag(loss, diagonal=0)  # in fact, the elements in the diagonal is the loss
            loss = torch.sum(loss)

            # average the loss
            loss = loss / float(batch_size)

        return loss












