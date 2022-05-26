from advent.model.deeplabv2 import Bottleneck, ResNetMulti

AFFINE_PAR = True
from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np


    
class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, size):

        x = self.relu(self.bn1(self.conv1(x)))
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)

        return x

class ResNetMultiDepth(ResNetMulti):
    def __init__(self, block, layers, num_classes, multi_level):
        super().__init__(block, layers, num_classes, multi_level)
        self.dec4 = nn.Conv2d(64, 2048, kernel_size=1, stride=1, padding=0, bias=True)  # enlaishi 128,2048
        self.dec4.weight.data.normal_(0, 0.01)
        block_channel = [256, 512, 1024, 2048]
        num_features = 64
        num_output_features = 16

        self.att1 = self.attention(block_channel[0], block_channel[0] // 16)
        self.att2 = self.attention(block_channel[1], block_channel[1] // 16)
        self.att3 = self.attention(block_channel[2], block_channel[2] // 16)
        self.att4 = self.attention(block_channel[3], block_channel[3] // 16)
        self.up1 = _UpProjection(block_channel[0], num_output_features)
        self.up2 = _UpProjection(block_channel[1], num_output_features)
        self.up3 = _UpProjection(block_channel[2], num_output_features)
        self.up4 = _UpProjection(block_channel[3], num_output_features)
        self.mff_conv = nn.Conv2d(num_features, num_features,
                                  kernel_size=5, stride=1, padding=2, bias=False)
        self.mff_bn = nn.BatchNorm2d(num_features)

        # R
        self.convR0 = nn.Conv2d(num_features, num_features,
                                kernel_size=5, stride=1, padding=2, bias=False)
        self.bnR0 = nn.BatchNorm2d(num_features)
        self.convR2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def attention(self, features1, features2):
        prior = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        conv1 = nn.Conv2d(features1, features2, kernel_size=1, bias=False)
        # bn = nn.BatchNorm2d(features)
        relu = nn.ReLU()
        conv2 = nn.Conv2d(features2, features1, kernel_size=1, bias=False)
        sigmoid = nn.Sigmoid()
        return nn.Sequential(prior, conv1, relu, conv2, sigmoid)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        x4 = x_block4
        if self.multi_level:
            seg_conv4 = self.layer5(x)
        else:
            seg_conv4 = None

        x_att1 = self.att1(x_block1)
        x_att2 = self.att2(x_block2)
        x_att3 = self.att3(x_block3)
        x_att4 = self.att4(x_block4)
        x_m1 = self.up1(x_block1 * x_att1, [x_block2.size(2), x_block2.size(3)])
        x_m2 = self.up2(x_block2 * x_att2, [x_block2.size(2),x_block2.size(3)])
        x_m3 = self.up3(x_block3 * x_att3, [x_block2.size(2), x_block2.size(3)])
        x_m4 = self.up4(x_block4 * x_att4, [x_block2.size(2),x_block2.size(3)])
        x_mff = self.mff_bn(self.mff_conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x_mff = F.relu(x_mff)


        R0 = self.convR0(x_mff)
        R1 = self.bnR0(R0)
        R2 = F.relu(R1)
        out = self.convR2(R2)

        depth = F.upsample(out, size=[228, 304], mode='bilinear', align_corners=True)

        x4_dec = self.dec4(x_mff)
        x4_dec = self.relu(x4_dec)
        x4 = x4 * x4_dec
        seg_conv5 = self.layer6(x4)  # produce segmap 2
        return seg_conv4, seg_conv5, depth

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        b.append(self.att1.parameters())
        b.append(self.att2.parameters())
        b.append(self.att3.parameters())
        b.append(self.att4.parameters())
        b.append(self.up1.parameters())
        b.append(self.up2.parameters())
        b.append(self.up3.parameters())
        b.append(self.up4.parameters())
        b.append(self.mff_bn.parameters())
        b.append(self.mff_conv.parameters())
        b.append(self.convR0.parameters())
        b.append(self.convR2.parameters())
        b.append(self.bnR0.parameters())
        b.append(self.dec4.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

def get_deeplab_v2_depth(num_classes=16, multi_level=False):
    model = ResNetMultiDepth(
        Bottleneck, [3, 4, 23, 3], num_classes, multi_level
    )
    return model
