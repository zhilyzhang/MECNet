import torch
import torch.nn.functional as F
from torch import nn


def _gauss_1d_kernel(size, sigma):
    r'''
    args:
        size (int): the size of kernel
        sigma (float): sigma for normal distribution
    :return:
        torch.Tensor: 1D kernel (1 × 1 × size)
    '''
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def _gauss_2d_kernel(size, channel, sigma=1.5):
    r'''
    args:
        :param size: the size of 2D kernel
        :param sigma: sigma for normal distribution
    :return:
        torch.Tensor: 2D kernel (channel × 1 × size × size)
    '''
    _1D_kernel = _gauss_1d_kernel(size, sigma).squeeze(0)
    _2D_kernel = _1D_kernel.t().mm(_1D_kernel).unsqueeze(0).unsqueeze(0)
    k = _2D_kernel.expand(channel, 1, size, size).contiguous()
    return k


def _ssim(X, Y, window_size, size_average=True, K=(0.01, 0.03)):
    r'''
    args
        :param X (torch.Tensor): images
        :param Y (torch.Tensor): images
        :param window_size (int):
        :param size_average: if size_average=True, ssim of all images will be averaged as a scalar
    :return:
        torch.Tensor: ssim results
    '''
    C = X.shape[1]
    window = _gauss_2d_kernel(window_size, C).to(X.device)
    mu1 = F.conv2d(X, window, padding=window_size//2, stride=1, groups=C)
    mu2 = F.conv2d(Y, window, padding=window_size//2, stride=1, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1 = F.conv2d(X * X, window, padding=window_size//2, stride=1, groups=C) - mu1_sq
    sigma2 = F.conv2d(Y * Y, window, padding=window_size//2, stride=1, groups=C) - mu2_sq
    sigma12 = F.conv2d(X * Y, window, padding=window_size//2, stride=1, groups=C) - mu1_mu2

    c1, c2 = K
    ssim = (2*mu1_mu2 + c1) * (2*sigma12 + c2) / ((mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2))

    if size_average:
        return ssim.mean()
    else:
        return ssim.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    def __init__(self, win_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.win_size = win_size
        self.size_average = size_average

    def forward(self, x, y, is_sigmoid=False):
        y = y.float()
        if is_sigmoid:
            x = torch.sigmoid(x)

        ssim = _ssim(
            X=x,
            Y=y,
            window_size=self.win_size,
            size_average=self.size_average
        )
        return 1 - ssim


if __name__ == '__main__':
    # res = _gauss_1d_kernel(3, sigma=1.5)
    # res = _gauss_2d_kernel(3, channel=1, sigma=1.5)
    # print(res, res.sum())
    X = torch.randn(1, 1, 32, 32)
    Y = X + 0.01
    res = _ssim(X, Y, 11, size_average=False)
    # ssim = SSIM(11, size_average=False)
    # res = ssim(X, Y)
    print(res)

