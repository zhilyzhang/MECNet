import os
import numpy as np
from glob import glob
import cv2 as cv
import json
from shutil import move
from tqdm import tqdm
from collections import Counter


def cal_confu_matrix(label, predict, class_num=2):
    confu_list = []
    for i in range(class_num):
        c = Counter(predict[np.where(label == i)])
        single_row = []
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int32)


def metrics(confu_mat_total):
    '''
    :param confu_mat: 总的混淆矩阵
    backgound：是否干掉背景
    :return: excel写出混淆矩阵, precision，recall，IOU，f-score
    FinalClass,False表示去掉最后一个类别，计算mIou, mf-score
    '''
    class_num = confu_mat_total.shape[0]

    confu_mat = confu_mat_total.astype(np.float32) + 0.0001
    col_sum = np.sum(confu_mat, axis=1)  # 按行求和
    raw_sum = np.sum(confu_mat, axis=0)  # 按列求和

    '''计算各类面积比，以求OA值'''
    oa = 0
    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()

    # 将混淆矩阵写入excel中
    TP = []  # 识别中每类分类正确的个数

    for i in range(class_num):
        TP.append(confu_mat[i, i])

    # 计算f1-score
    TP = np.array(TP)
    FN = col_sum - TP
    FP = raw_sum - TP

    # 计算并写出precision，recall, IOU
    precision = TP[1] / col_sum[1]
    recall = TP[1] / raw_sum[1]
    iou = TP[1] / (TP[1] + FP[1] + FN[1])

    return oa, precision, recall, iou


def get_cal_metrics(pred_path, label_path):
    confu_matrix = np.zeros((2, 2), dtype=np.int32)
    pred_pathes = glob(pred_path + '/*.tif')
    for path in pred_pathes:
        basename = os.path.basename(path)
        pred = cv.imread(path, cv.IMREAD_GRAYSCALE)
        label = cv.imread(label_path + '/' + basename, cv.IMREAD_GRAYSCALE)
        pred[pred == 255] = 1
        label[label == 255] = 1
        confu_matrix += cal_confu_matrix(label, pred)
    oa, precision, recall, iou = metrics(confu_matrix)
    print('pixel acc: %.3f%%' % (oa*100))
    print('precision: %.3f%%' % (precision*100))
    print('recall: %.3f%%' % (recall*100))
    print('IOU: %.3f%%' % (iou*100))
    # return oa, precision, recall, iou


if __name__ == '__main__':
    pred_path = '/home/zzl/datasets/NewZealand-building-datset/res/transunet/results'
    # pred_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\experimental_results\MFCNetV2\predict'
    label_path = '/home/zzl/datasets/NewZealand-building-datset/test/label'
    get_cal_metrics(pred_path, label_path)

