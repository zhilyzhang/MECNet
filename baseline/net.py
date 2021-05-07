import torch
from torch import nn
from math import sqrt
import warnings
import torch.nn.functional as F
from functools import partial
from torchvision import models
import cv2 as cv
import numpy as np
from torch.autograd import Variable
import os
from baseline.seg_hrnet import Bottleneck


def conv3x3_bn_relu(in_features, out_features):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True)
    )


def output_layer(in_features, class_num):
    return nn.Conv2d(in_features, class_num, kernel_size=1, stride=1)


class _PAM(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PAM, self).__init__()
        reduceRate = 8
        self.conv_b = nn.Conv2d(in_channels, in_channels // reduceRate, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // reduceRate, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _CAM(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_CAM, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class _GAM(nn.Module):
    # Channel Merge Attention
    def __init__(self, in_channels):
        super(_GAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1_1 = nn.Conv2d(
            in_channels, in_channels//2, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.conv1_2 = nn.Conv2d(
            in_channels//2, in_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # global average pool
        x1 = self.avg_pool(x)
        x1 = self.conv1_1(x1)
        x1 = self.relu(x1)
        x1 = self.conv1_2(x1)
        x1 = self.sigmoid(x1)  # output N * C ?
        x2 = x * x1
        x2 = x + x2
        x2 = self.conv2(x2)  # 增加的
        return x2


class _DAHead(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PAM(inter_channels, **kwargs)
        self.cam = _CAM(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        fusion_out = self.out(feat_fusion)

        return fusion_out


class _TAHead(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_TAHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv_p1 = conv3x3_bn_relu(in_channels, inter_channels)
        self.conv_p2 = conv3x3_bn_relu(inter_channels, in_channels)

        self.conv_c1 = conv3x3_bn_relu(in_channels, inter_channels)
        self.conv_c2 = conv3x3_bn_relu(inter_channels, in_channels)

        self.conv_g1 = conv3x3_bn_relu(in_channels, inter_channels)
        self.conv_g2 = conv3x3_bn_relu(inter_channels, in_channels)

        self.pam = _PAM(inter_channels, **kwargs)
        self.cam = _CAM(**kwargs)
        self.gam = _GAM(inter_channels)

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_g = self.conv_g1(x)
        feat_g = self.gam(feat_g)
        feat_g = self.conv_g2(feat_g)

        feat_fusion = feat_p + feat_c + feat_g

        fusion_out = self.out(feat_fusion)

        return fusion_out


def conv1x1_bn_relu(in_features, out_features):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=1),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True)
    )


class NetFW(nn.Module):
    def __init__(self, class_num=1):
        super(NetFW, self).__init__()
        self.class_num = class_num

        channels = [64, 128, 256, 512]
        self.encoder1 = nn.Sequential(
            conv3x3_bn_relu(3, channels[0]),
            Bottleneck(channels[0], planes=channels[0]//4),
            # conv3x3_bn_relu(channels[0], channels[0]),
        )

        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            conv3x3_bn_relu(channels[0], channels[1]),
            Bottleneck(channels[1], planes=channels[1] // 4),
            # conv3x3_bn_relu(channels[1], channels[1])
        )

        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            conv3x3_bn_relu(channels[1], channels[2]),
            Bottleneck(channels[2], planes=channels[2] // 4),
            Bottleneck(channels[2], planes=channels[2] // 4),
            # conv3x3_bn_relu(channels[2], channels[2]),
            # conv3x3_bn_relu(channels[2], channels[2]),
        )

        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            conv3x3_bn_relu(channels[2], channels[3]),
            Bottleneck(channels[3], planes=channels[3] // 4),
            Bottleneck(channels[3], planes=channels[3] // 4)
        )

        self.TAM = _TAHead(channels[3])
        self.out4 = output_layer(channels[3], class_num=self.class_num)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv1x1_bn_relu(channels[3], channels[2])
        )
        self.decoder3 = nn.Sequential(
            conv1x1_bn_relu(channels[2] * 2, channels[2]),
            Bottleneck(channels[2], planes=channels[2] // 4),
            conv3x3_bn_relu(channels[2], channels[2]), )
        self.out3 = output_layer(channels[2], class_num=self.class_num)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv1x1_bn_relu(channels[2], channels[1]))

        self.decoder2 = nn.Sequential(
            conv1x1_bn_relu(channels[1] * 2, channels[1]),
            Bottleneck(channels[1], planes=channels[1] // 4),
            conv3x3_bn_relu(channels[1], channels[1]), )
        self.out2 = output_layer(channels[1], class_num=self.class_num)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv1x1_bn_relu(channels[1], channels[0]))

        self.decoder1 = nn.Sequential(
            conv1x1_bn_relu(channels[0] * 2, channels[0]),
            Bottleneck(channels[0], planes=channels[0] // 4),
            conv3x3_bn_relu(channels[0], channels[0]), )
        self.num_features = channels[0]
        self.AFC = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        tam = self.TAM(e4)
        out4 = nn.Upsample(scale_factor=8)(self.out4(tam))

        up3 = self.up3(tam)
        d3 = self.decoder3(torch.cat((up3, e3), dim=1))
        out3 = nn.Upsample(scale_factor=4)(self.out3(d3))

        up2 = self.up2(d3)
        d2 = self.decoder2(torch.cat((up2, e2), dim=1))
        out2 = nn.Upsample(scale_factor=2)(self.out2(d2))

        up1 = self.up1(d2)
        d1 = self.decoder1(torch.cat((up1, e1),dim=1))

        out1 = self.AFC(d1)

        return out1, out2, out3, out4

