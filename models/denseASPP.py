import torch
import torch.nn.functional as F
from math import sqrt
from torch import nn
from collections import OrderedDict
from torch.nn import BatchNorm2d as bn
from cfgs import DenseASPP201, DenseASPP121


class DenseASPP(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """
    def __init__(self, model_cfg, n_class=5, output_stride=8):
        super(DenseASPP, self).__init__()
        bn_size = model_cfg['bn_size']
        drop_rate = model_cfg['drop_rate']
        growth_rate = model_cfg['growth_rate']
        num_init_features = model_cfg['num_init_features']
        block_config = model_cfg['block_config']

        dropout0 = model_cfg['dropout0']
        dropout1 = model_cfg['dropout1']
        d_feature0 = model_cfg['d_feature0']
        d_feature1 = model_cfg['d_feature1']

        feature_size = int(output_stride / 8)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', bn(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            # ('relu0', Mish()),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        # block1*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % 1, block)
        num_features = num_features + block_config[0] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features.add_module('transition%d' % 1, trans)
        num_features = num_features // 2

        # block2*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % 2, block)
        num_features = num_features + block_config[1] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=feature_size)
        self.features.add_module('transition%d' % 2, trans)
        num_features = num_features // 2

        # block3*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate, dilation_rate=int(2 / feature_size))
        self.features.add_module('denseblock%d' % 3, block)
        num_features = num_features + block_config[2] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=1)
        self.features.add_module('transition%d' % 3, trans)
        num_features = num_features // 2

        # block4*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate, dilation_rate=int(4 / feature_size))
        self.features.add_module('denseblock%d' % 4, block)
        num_features = num_features + block_config[3] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=1)
        self.features.add_module('transition%d' % 4, trans)
        num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5-', bn(num_features))
        if feature_size > 1:
            self.features.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))

        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=num_features + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)
        num_features = num_features + 5 * d_feature1
        self.num_features = num_features
        self.classification = nn.Sequential(
            nn.Dropout2d(p=dropout1),
            nn.Conv2d(in_channels=self.num_features, out_channels=n_class, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode='bilinear'),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, _input):
        feature = self.features(_input)

        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)

        cls = self.classification(feature)

        return cls


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm1', bn(input_num, momentum=0.0003)),

        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', bn(num1, momentum=0.0003)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation_rate=1):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', bn(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('relu1', Mish()),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', bn(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        # self.add_module('relu2', Mish()),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, dilation=dilation_rate, padding=dilation_rate, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dilation_rate=1):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate, dilation_rate=dilation_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride=2):
        super(_Transition, self).__init__()
        self.add_module('norm', bn(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('relu', Mish())
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if stride == 2:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=stride))


def denseASPP121(n_class=1):
    return DenseASPP(DenseASPP121.Model_CFG, n_class=n_class)


def denseASPP201():
    return DenseASPP(DenseASPP201.Model_CFG, n_class=1)


class UNetTAM_CMAmc(nn.Module):
    def __init__(self, class_num):
        '''
        TAM: three kinds of Attention for feature merge.
        CMA: channels merge for decode and encode part.
        '''
        super(UNetTAM_CMAmc, self).__init__()
        self.class_num = class_num

        channels = [32, 64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            Conv2dRelu(3, channels[0]),
            Conv2dBNRelu(channels[0], channels[0]),)

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[0], channels[1]),
            Conv2dBNRelu(channels[1], channels[1]),)

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[1], channels[2]),
            Conv2dBNRelu(channels[2], channels[2]),
            Conv2dBNRelu(channels[2], channels[2]),)

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[2], channels[3]),
            Conv2dBNRelu(channels[3], channels[3]),
            Conv2dBNRelu(channels[3], channels[3]),)

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[3], channels[4]),
            Conv2dBNRelu(channels[4], channels[4]),
            Conv2dBNRelu(channels[4], channels[4]),)

        self.TAM = _TAHead(channels[4])
        # 16
        self.out5 = nn.Sequential(
            Conv2dRelu(channels[4], channels[4]//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(channels[4]//2, self.class_num, kernel_size=1, stride=1)
        )

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            CMAD(channels[3], kernel_size=(3, 3)),
            Conv2dBNRelu(channels[3], channels[3]),
        )
        # 32
        self.out4 = nn.Sequential(
            Conv2dRelu(channels[3], channels[3] // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(channels[3] // 2, self.class_num, kernel_size=1, stride=1)
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            CMAD(channels[2], kernel_size=(5, 5)),
            Conv2dBNRelu(channels[2], channels[2]),
        )
        # 64
        self.out3 = nn.Sequential(
            Conv2dRelu(channels[2], channels[2] // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(channels[2] // 2, self.class_num, kernel_size=1, stride=1)
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            CMAD(channels[1], kernel_size=(5, 5)),
            Conv2dBNRelu(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            # Conv2dBNRelu(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
        )
        # 128
        self.out2 = nn.Sequential(
            Conv2dRelu(channels[1], channels[1] // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(channels[1] // 2, self.class_num, kernel_size=1, stride=1)
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            CMAD(channels[0], kernel_size=(5, 5)),
            Conv2dBNRelu(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # con2_no_grad = conv2.detach()
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        tam = self.TAM(conv5)
        out5 = self.out5(tam)

        deconv4 = self.deconv4(tam)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)
        out4 = self.out4(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)
        out3 = self.out3(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)
        out2 = self.out2(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        out1 = self.conv9(conv9)

        return out1, out2, out3, out4, out5



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


def conv3x3_bn_relu(in_features, out_features):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True)
    )


def conv1x1_bn_relu(in_features, out_features):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=1),
        nn.BatchNorm2d(out_features),
        nn.ReLU(inplace=True)
    )


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


class BigConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super().__init__()
        pad0 = kernel_size[0] // 2  # 填充大小跟卷积核大小有关
        pad1 = kernel_size[1] // 2

        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv3 = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1)

        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        # combine two paths
        x = x_l + x_r

        x = self.conv3(x)  # 加强信息融合
        return self.bn(self.relu(x))


class CMA(nn.Module):
    # channels merge Attention for decode and encode part.
    # 不考虑直接相加，高层语义和低级；两种融合优势
    def __init__(self, in_channels, kernel_size=(7, 7)):
        super(CMA, self).__init__()
        reduceRate = 2
        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.conv_w = nn.Sequential(
            BigConvBlock(in_channels, in_channels // reduceRate, kernel_size),
            nn.Conv2d(in_channels // reduceRate, in_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.conv_c(x)

        b = self.conv_w(x) * self.beta

        x = x * b + x
        return x


class CMAD(nn.Module):
    # channels merge Attention for decode and encode part.
    # 不考虑直接相加，高层语义和低级；两种融合优势
    def __init__(self, in_channels, kernel_size=(5, 5)):
        super(CMAD, self).__init__()
        reduceRate = 2
        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.conv_w = nn.Sequential(
            BigConvBlock(in_channels, in_channels // reduceRate, kernel_size),
            nn.Conv2d(in_channels // reduceRate, in_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.gam = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.beta = nn.Parameter(torch.zeros(1))
        self.alfa = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.conv_c(x)

        b = self.conv_w(x) * self.beta
        a = self.gam(x) * self.alfa

        x = x * a + x * b + x
        return x


def Conv2dRelu(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU()  # inplace=True
    )


def Conv2dBNRelu(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU()  # inplace=True
    )

if __name__ == "__main__":
    # b_size = 1
    # c = 4
    # m, n = 2, 2
    # e = torch.arange(b_size*c*m*n).reshape(b_size, c, m, n)
    # d = torch.arange(b_size*c*m*n).reshape(b_size, c, m, n)
    # print(e)
    # print(d)
    # e = torch.chunk(e, c//2, dim=1)
    # d = torch.chunk(d, c//2, dim=1)
    # x = torch.cat(([torch.cat((e_, d_), dim=1) for e_, d_ in zip(e, d)]), dim=1)
    # x = nn.PixelShuffle(upscale_factor=2)(x)
    # print(x)
    from thop import profile
    import torchsummary

    band_num = 3
    class_num = 1
    model = denseASPP121()
    input = torch.randn(1, 3, 512, 512)
    # xs = model(input)
    flops, params = profile(model, inputs=(input,))
    # flops, params = profile(model, inputs=(1, band_num, 256, 256))
    model.cuda()
    torchsummary.summary(model, (band_num, 512, 512))
    print('flops(G): %.3f' % (flops / 1e+9))
    print('params(M): %.3f' % (params / 1e+6))
