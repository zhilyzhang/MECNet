import numpy as np


def label2RGB(label):
    '''
    居民地 1 红色(255,0,0)
    道路 2 绿色(0,255,0)
    水体 3 蓝色(0,0,255)
    植被 4 黄色(255,255,0)
    其它类 0 黑色(0,0,0)
    '''
    m, n = label.shape
    rgb = np.zeros((m, n, 3), dtype=np.uint8)
    rgb[label == 1] = np.array((0, 0, 255), dtype=np.uint8)  # BGR
    rgb[label == 2] = np.array((0, 255, 0), dtype=np.uint8)
    rgb[label == 3] = np.array((255, 0, 0), dtype=np.uint8)
    rgb[label == 4] = np.array((0, 255, 255), dtype=np.uint8)
    return rgb


def RGBtolabel2015(img):
    m, n, _ = img.shape
    label = np.zeros((m, n), dtype=np.int)

    tmp = (img == np.array([0, 248, 248], dtype=np.uint8)).astype(dtype=np.int32)  # BGR
    label[np.sum(tmp, axis=2) == 3] = 1

    tmp = (img == np.array([248, 0, 0], dtype=np.uint8)).astype(dtype=np.int32)
    label[np.sum(tmp, axis=2) == 3] = 2

    tmp = (img == np.array([0, 0, 248], dtype=np.uint8)).astype(dtype=np.int32)
    label[np.sum(tmp, axis=2) == 3] = 3

    tmp = (img == np.array([0, 248, 0], dtype=np.uint8)).astype(dtype=np.int32)
    label[np.sum(tmp, axis=2) == 3] = 4
    return label