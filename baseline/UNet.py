from torch import nn
import torch
import torchsummary
from baseline.seg_hrnet import Bottleneck


def Conv2dRelu(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU()  # inplace=True
    )


def Conv2dBNRelu(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)  #
    )


def upsample_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(in_channels, out_channels, kernel_size=1))


class UNet(nn.Module):
    def __init__(self, class_num=1, band_num=3):
        super(UNet, self).__init__()
        self.band_num = band_num
        self.class_num = class_num

        # channels = [32, 64, 128, 256, 512]
        channels = [64, 128, 256, 512, 512]
        self.conv1 = nn.Sequential(
            Conv2dBNRelu(self.band_num, channels[0]),
            Conv2dBNRelu(channels[0], channels[0]),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[0], channels[1]),
            Conv2dBNRelu(channels[1], channels[1]),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[1], channels[2]),
            Conv2dBNRelu(channels[2], channels[2]),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[2], channels[3]),
            Conv2dBNRelu(channels[3], channels[3]),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[3], channels[4]),
            Conv2dBNRelu(channels[4], channels[4]),
        )

        self.deconv4 = upsample_layer(channels[4], channels[3])
        self.conv6 = nn.Sequential(
            Conv2dBNRelu(channels[4]*2, channels[3]),
            Conv2dBNRelu(channels[3], channels[3]),
        )

        self.deconv3 = upsample_layer(channels[3], channels[2])
        self.conv7 = nn.Sequential(
            Conv2dBNRelu(channels[3], channels[2]),
            Conv2dBNRelu(channels[2], channels[2]),
        )

        self.deconv2 = upsample_layer(channels[2], channels[1])
        self.conv8 = nn.Sequential(
            Conv2dBNRelu(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
            Conv2dBNRelu(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
        )

        self.deconv1 = upsample_layer(channels[1], channels[0])
        self.conv9 = nn.Sequential(
            Conv2dBNRelu(channels[1], channels[0], kernel_size=3, stride=1, padding=1),
            Conv2dBNRelu(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        output = self.conv10(conv9)

        return output


class UNetSpalt(nn.Module):
    def __init__(self, class_num=1, band_num=3):
        super(UNetSpalt, self).__init__()
        self.band_num = band_num
        self.class_num = class_num

        channels = [32, 64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            Conv2dRelu(self.band_num, channels[0]),
            Bottleneck(channels[0], planes=channels[0]//4),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[0], channels[1]),
            Bottleneck(channels[1], planes=channels[1]//4),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[1], channels[2]),
            Bottleneck(channels[2], planes=channels[2]//4),
            Bottleneck(channels[2], planes=channels[2]//4),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[2], channels[3]),
            Bottleneck(channels[3], planes=channels[3]//4),
            Bottleneck(channels[3], planes=channels[3]//4),
            # Conv2dBNRelu(channels[3], channels[3]),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[3], channels[4]),
            Bottleneck(channels[4], planes=channels[4] // 4),
            Bottleneck(channels[4], planes=channels[4] // 4),
            # Conv2dBNRelu(channels[4], channels[4]),
        )

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            Conv2dBNRelu(channels[4], channels[3]),
            Bottleneck(channels[3], planes=channels[3] // 4),
            # Conv2dBNRelu(channels[3], channels[3]),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            Conv2dBNRelu(channels[3], channels[2]),
            Bottleneck(channels[2], planes=channels[2] // 4),
            # Conv2dBNRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            Conv2dBNRelu(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
            Bottleneck(channels[1], planes=channels[1] // 4),
            # Conv2dBNRelu(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            Conv2dBNRelu(channels[1], channels[0], kernel_size=3, stride=1, padding=1),
            Bottleneck(channels[0], planes=channels[0] // 4),
            # Conv2dBNRelu(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
        )

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        output = self.conv10(conv9)

        return output


class UNetSpaltV2(nn.Module):
    def __init__(self, class_num=1, band_num=3):
        super(UNetSpaltV2, self).__init__()
        self.band_num = band_num
        self.class_num = class_num

        channels = [32, 64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            Conv2dRelu(self.band_num, channels[0]),
            Bottleneck(channels[0], planes=channels[0]//4),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[0], channels[1]),
            Bottleneck(channels[1], planes=channels[1]//4),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[1], channels[2]),
            Bottleneck(channels[2], planes=channels[2]//4),
            Bottleneck(channels[2], planes=channels[2]//4),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[2], channels[3]),
            Bottleneck(channels[3], planes=channels[3]//4),
            Bottleneck(channels[3], planes=channels[3]//4),
            # Conv2dBNRelu(channels[3], channels[3]),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2dBNRelu(channels[3], channels[4]),
            Bottleneck(channels[4], planes=channels[4] // 4),
            Bottleneck(channels[4], planes=channels[4] // 4),
            # Conv2dBNRelu(channels[4], channels[4]),
        )

        self.deconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.Sequential(
            Conv2dBNRelu(channels[4], channels[3]),
            # Bottleneck(channels[3], planes=channels[3] // 4),
            Conv2dBNRelu(channels[3], channels[3]),
        )

        self.deconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.Sequential(
            Conv2dBNRelu(channels[3], channels[2]),
            # Bottleneck(channels[2], planes=channels[2] // 4),
            Conv2dBNRelu(channels[2], channels[2]),
        )

        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.Sequential(
            Conv2dBNRelu(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
            # Bottleneck(channels[1], planes=channels[1] // 4),
            Conv2dBNRelu(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
        )

        self.deconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.Sequential(
            Conv2dBNRelu(channels[1], channels[0], kernel_size=3, stride=1, padding=1),
            # Bottleneck(channels[0], planes=channels[0] // 4),
            Conv2dBNRelu(channels[0], channels[0], kernel_size=3, stride=1, padding=1))

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.deconv4(conv5)
        conv6 = torch.cat((deconv4, conv4), 1)
        conv6 = self.conv6(conv6)

        deconv3 = self.deconv3(conv6)
        conv7 = torch.cat((deconv3, conv3), 1)
        conv7 = self.conv7(conv7)

        deconv2 = self.deconv2(conv7)
        conv8 = torch.cat((deconv2, conv2), 1)
        conv8 = self.conv8(conv8)

        deconv1 = self.deconv1(conv8)
        conv9 = torch.cat((deconv1, conv1), 1)
        conv9 = self.conv9(conv9)

        output = self.conv10(conv9)

        return output


if __name__ == '__main__':
    '''
    UNet
    '''
    from thop import profile
    import torchsummary

    band_num = 3
    class_num = 1
    model = UNet()
    input = torch.randn(1, 3, 512, 512)
    flops, params = profile(model, inputs=(input,))
    # flops, params = profile(model, inputs=(1, band_num, 256, 256))
    model.cuda()
    torchsummary.summary(model, (band_num, 512, 512))
    print('flops(G): %.3f' % (flops / 1e+9))
    print('params(M): %.3f' % (params / 1e+6))
