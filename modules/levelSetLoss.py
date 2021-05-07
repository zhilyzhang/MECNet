import torch
from torch import nn
import torch.nn.functional as F


def Cal_phi(tensor):
    '''
    args:
        :param tensor: output for CNN
    :return: sigmoid(output) - 0.5
    '''
    return torch.sigmoid(tensor) - 0.5


def H_epsilon(z, epsilon=0.05):
    return 1 / 2 * (1 + torch.tanh(z / epsilon))


class BinaryLevelSetLoss(nn.Module):
    def __init__(self, ):
        super(BinaryLevelSetLoss, self).__init__()

    def forward(self, pred, gt):
        phi = Cal_phi(pred)
        he = H_epsilon(phi)
        c1 = torch.sum(gt * he) / he.sum()
        c2 = torch.sum(gt * (1-he)) / torch.sum(1-he)

        loss = torch.sum(torch.pow((gt-c1), 2)*he) + torch.sum(torch.pow((gt-c2), 2)*(1 - he))
        return loss


def active_contour_loss(y_true, y_pred, weight=10):
    '''
    y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
    weight: scalar, length term weight.
    '''
    # length term
    delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
    delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

    delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
    delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
    delta_pred = torch.abs(delta_r + delta_c)

    epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
    lenth = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

    # region term
    c_in = torch.ones_like(y_pred)
    c_out = torch.zeros_like(y_pred)

    region_in = torch.mean(y_pred * (y_true - c_in) ** 2)  # equ.(12) in the paper, mean is used instead of sum.
    region_out = torch.mean((1 - y_pred) * (y_true - c_out) ** 2)
    region = region_in + region_out

    loss = weight * lenth + region

    return loss


class LSLoss(nn.Module):
    def __init__(self):
        super(LSLoss, self).__init__()

    def forward(self, y_true, y_pred, weight=10):
        loss = active_contour_loss(y_true, y_pred, weight=weight)
        return loss


if __name__ == '__main__':
    a = torch.randn(1, 1, 128, 128)
    b = torch.sigmoid(a)
    # print(b)
    LSloss = LSLoss()
    loss = LSloss(b, b)
    print(loss)
