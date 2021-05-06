import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torch import nn


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    # print(window_size.shape, channel)
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        # print(self.window.shape)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window

        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


# class IOULoss(torch.nn.Module):
#     def __init__(self, size_average=True):
#         super(IOULoss, self).__init__()
#         # 255 归为0
#         self.size_average = size_average
#
#     def forward(self, predict, label):
#         tmp = predict * label
#         loss = (
#                 1-(tmp.sum(-1).sum(-1) /
#                    (predict.sum(-1).sum(-1) + label.sum(-1).sum(-1) - tmp.sum(-1).sum(-1) + 1e-12)).mean(dim=1)).mean()
#         print(loss.shape)
#         return loss


class RecallLoss(nn.Module):
    def __init__(self):
        super(RecallLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.float()
        res = y_true * y_pred
        return 1 - (res.sum() / (y_true.sum() + 1e-6))


class PrecisionLoss(nn.Module):
    def __init__(self):
        super(PrecisionLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.float()
        res = y_true * y_pred
        return 1 - (res.sum() / (y_pred.sum() + 1e-6))


class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.float()
        res = y_true * y_pred
        return 1 - (res.sum() / (y_true.sum() + y_pred.sum() - res.sum() + 1e-6))


# def cal_iou(y_pred, y_true):


def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean'else loss.sum() if reduction=='sum'else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes=1, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, label):
        '''
        :param pred: B×Num_classes×M×N
        :param label: B×M×N
        :return:
        '''
        assert self.num_classes == pred.size(1)
        true_dist = label.data.clone()
        true_dist.fill_(self.smoothing)
        label = label.long()
        true_dist.scatter_(1, label.data, self.confidence)
        return self.criterion(pred.squeeze(1), true_dist)

# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self, epsilon:float=0.1, reduction='mean'):
#         super().__init__()
#         self.epsilon = epsilon
#         self.reduction = reduction
#
#     def forward(self, preds, target):
#         n = preds.size()[-1]
#         log_preds = F.log_softmax(preds, dim=-1)
#         loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
#         nll = F.nll_loss(log_preds, target, reduction=self.reduction)
#         return linear_combination(loss/n, nll, self.epsilon)


if __name__ == '__main__':
    # size = 4
    # pred = torch.rand(1, 1, size, size)
    # label = pred.squeeze(1)
    # print(pred)
    # label[label>0.5] = 1.0
    # label[label<0.5] = 0
    # lsce = LabelSmoothingCrossEntropy()
    # loss = lsce(pred, label)
    # print(loss)

    create_window(11, 1)
