import torch
from torch import nn
from torch.nn.modules.utils import _pair
from dropblock import DropBlock2D
import torch.nn.functional as F


class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg

        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        self.conv = nn.Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                           groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        # print(x.shape)  #torch.Size([2, 64, 64, 64])
        x = self.conv(x)  # 进行1×1的c'/k/r的卷积

        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)
        # print(x.shape) # torch.Size([2, 128, 64, 64])
        batch, rchannel = x.shape[:2]
        if self.radix > 1:  # 分成radix组，进行split-attention
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)   # torch.Size([2, 64, 64, 64])
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)  # torch.Size([2, 64, 1, 1])
        gap = self.fc1(gap)  # torch.Size([2, 32, 1, 1])
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        # print(atten.shape)  # torch.Size([2, 128, 1, 1])
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        # print(atten.shape) # torch.Size([2, 128, 1, 1])

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        # print(out.shape)  # torch.Size([2, 64, 64, 64])
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=2, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 norm_layer=nn.BatchNorm2d, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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
