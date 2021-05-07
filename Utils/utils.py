import numpy as np
import os
from glob import glob
import cv2 as cv
from torch.optim.lr_scheduler import _LRScheduler
import torch


def get_bin_cls_iou(pred, label):
    '''
    用于二分类计算IOU
    :param pred: B×M×N
    :param label: B×M×N
    :return: i_c, u_c
    '''
    tmp_ic = pred * label
    i_c = tmp_ic.sum()

    tmp_uc = pred + label
    u_c = tmp_uc.sum() - i_c
    return i_c, u_c


def get_mul_cls_iou(pred, label, num_classes=2):
    '''
    用于多分类计算IOU
    :param pred: B×M×N
    :param label: B×M×N
    :param num_classes:
    :return: i_c, u_c numpy array: [num_classes, ]
    '''
    ic = np.zeros((num_classes, ))
    uc = np.zeros((num_classes, ))
    for i in range(num_classes):
        tmp_pred = np.where(pred==i, 1, 0)
        tmp_label = np.where(label==i, 1, 0)
        ic[i], uc[i] = get_bin_cls_iou(tmp_pred, tmp_label)
    return ic, uc


class PolyLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_iter, power, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        self.last_epoch = last_epoch
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch/self.max_iter) ** self.power for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, end_epoch, last_epoch=-1):
        self.end_epoch = end_epoch
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr / self.end_epoch * (self.last_epoch + 1)
                for base_lr in self.base_lrs]


class WarmUpStepLR(_LRScheduler):
    def __init__(self, optimizer, warm_up_end_epoch, step_size, gamma=0.1, last_epoch=-1):
        self.warm_up_end_epoch = warm_up_end_epoch
        self.step_size = step_size
        self.gamma = gamma
        super(WarmUpStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr / self.warm_up_end_epoch * (self.last_epoch)
                if self.last_epoch < self.warm_up_end_epoch
                else base_lr * self.gamma ** ((self.last_epoch - self.warm_up_end_epoch) // self.step_size)
                for base_lr in self.base_lrs]


def save_model(state, directory='./checkpoints', filename=None):
    if os.path.isdir(directory):
        pkl_filename = os.path.join(directory, filename)
        torch.save(state, pkl_filename)
        print('Save "{:}" in {:} successful'.format(pkl_filename, directory))
    else:
        print(' "{:}" directory is not exsits!'.format(directory))


def cal_acc(pred, gt):
    '''
    :param pred:
    :param gt:
    :return:
    '''
    tp = pred * gt
    tp[tp < 0.5] = 0
    tp[tp >= 0.5] = 1
    fm = pred + gt
    fm[fm > 1] = 1
    fm[fm < 0.5] = 0
    return np.sum(tp) / (np.sum(fm) + 1e-6)


if __name__ == '__main__':
    pred_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\val_data\label8'
    label_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\val_data\label'
    tifs = glob(os.path.join(pred_path, '*.tif'))
    ic, uc = 0, 0
    for tif_path in tifs:
        pred = cv.imread(tif_path, cv.IMREAD_GRAYSCALE)
        pred[pred == 255] = 1
        label = cv.imread(os.path.join(label_path, os.path.basename(tif_path)), cv.IMREAD_GRAYSCALE)
        label[label == 255] = 1
        tmp_ic, tmp_uc = get_bin_cls_iou(pred, label)
        ic += tmp_ic
        uc += tmp_uc
    print('%.3f%%' % (ic / uc * 100))

