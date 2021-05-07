import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch import nn
import torch
import warnings
from torch.autograd import Variable
import numpy as np
import cv2 as cv
from data_load import LoadTest
from glob import glob
from torch.nn import functional as F
from shutil import copyfile
from scipy import misc
from baseline.networks import PyAtNet
from baseline.UNet import UNet
from models.DeepLabV3_plus.deeplabv3_plus import DeepLabv3_plus
from PSPNet.pspnet import PSPNet
from models.danet import DANet
from baseline.fcn import FCN8s
from models.MECNet import *
from models.RefineNet.RefineNet import get_refinenet
from models.denseASPP import denseASPP121
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def predictWithOverlapB(model, img, patch_size=512, overlap_rate=1/4):
    '''

    :param model: a trained model
    :param img: a path for an image
    :param patch_size:
    :param overlap_rate:
    :return:
    '''
    # subsidiary value for the prediction of an image with overlap
    boder_value = int(patch_size * overlap_rate / 2)
    double_bv = boder_value * 2
    stride_value = patch_size - double_bv
    most_value = stride_value + boder_value

    # an image for prediction
    # img = cv.imread(img_path, cv.IMREAD_COLOR)
    m, n, _ = img.shape
    load_data = LoadTest()
    if max(m, n) <= patch_size:
        tmp_img = img
        tmp_img = load_data(tmp_img)
        with torch.no_grad():
            tmp_img = Variable(tmp_img)
            tmp_img = tmp_img.cuda().unsqueeze(0)
            result = model(tmp_img)
        output = result if not isinstance(result, (list, tuple)) else result[0]
        output = F.sigmoid(output)
        pred = output.data.cpu().numpy().squeeze(0).squeeze(0)  # [0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        return pred.astype(np.uint8)
    else:
        tmp = (m - double_bv) // stride_value  # 剔除重叠部分相当于无缝裁剪
        new_m = tmp if (m - double_bv) % stride_value == 0 else tmp + 1
        tmp = (n - double_bv) // stride_value
        new_n = tmp if (n - double_bv) % stride_value == 0 else tmp + 1
        FullPredict = np.zeros((m, n), dtype=np.uint8)
        for i in range(new_m):
            for j in range(new_n):
                if i == new_m - 1 and j != new_n - 1:
                    tmp_img = img[
                              -patch_size:,
                              j * stride_value:((j + 1) * stride_value + double_bv), :]
                elif i != new_m - 1 and j == new_n - 1:
                    tmp_img = img[
                              i * stride_value:((i + 1) * stride_value + double_bv),
                              -patch_size:, :]
                elif i == new_m - 1 and j == new_n - 1:
                    tmp_img = img[
                              -patch_size:,
                              -patch_size:, :]
                else:
                    tmp_img = img[
                              i * stride_value:((i + 1) * stride_value + double_bv),
                              j * stride_value:((j + 1) * stride_value + double_bv), :]
                tmp_img = load_data(tmp_img)
                with torch.no_grad():
                    tmp_img = Variable(tmp_img)
                    tmp_img = tmp_img.cuda().unsqueeze(0)
                    result = model(tmp_img)
                output = result if not isinstance(result, (list, tuple)) else result[0]
                output = F.sigmoid(output)
                pred = output.data.cpu().numpy().squeeze(0).squeeze(0)  # [0]

                pred[pred >= 0.5] = 255
                pred[pred < 0.5] = 0

                if i == 0 and j == 0:  # 左上角
                    FullPredict[0:most_value, 0:most_value] = pred[0:most_value, 0:most_value]
                elif i == 0 and j == new_n-1:  # 右上角
                    FullPredict[0:most_value, -most_value:] = pred[0:most_value, boder_value:]
                elif i == 0 and j != 0 and j != new_n - 1:  # 第一行
                    FullPredict[0:most_value, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                        pred[0:most_value, boder_value:most_value]

                elif i == new_m - 1 and j == 0:  # 左下角
                    FullPredict[-most_value:, 0:most_value] = pred[boder_value:, :-boder_value]
                elif i == new_m - 1 and j == new_n - 1:  # 右下角
                    FullPredict[-most_value:, -most_value:] = pred[boder_value:, boder_value:]
                elif i == new_m - 1 and j != 0 and j != new_n - 1:  # 最后一行
                    FullPredict[-most_value:, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                        pred[boder_value:, boder_value:-boder_value]

                elif j == 0 and i != 0 and i != new_m - 1:  # 第一列
                    FullPredict[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, 0:most_value] = \
                        pred[boder_value:-boder_value, 0:-boder_value]
                elif j == new_n - 1 and i != 0 and i != new_m - 1:  # 最后一列
                    FullPredict[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, -most_value:] = \
                        pred[boder_value:-boder_value, boder_value:]
                else:  # 中间情况
                    FullPredict[
                    boder_value + i * stride_value:boder_value + (i + 1) * stride_value,
                    boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                        pred[boder_value:-boder_value, boder_value:-boder_value]
        return FullPredict


class Test(object):
    def __init__(self, save_path, imgpath, weight_path):
        self.imgpath = imgpath
        self.weight_path = weight_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def predict(self):
        '''
        :return:
        '''
        img_pathes = glob(self.imgpath + '/*.tif')
        # model = get_refinenet(input_size=512, num_classes=1, pretrained=False)
        # model = DeepLabv3_plus(in_channels=3, num_classes=1, backend='resnet101', os=16)
        model = MECNet()
        # model = FCN()
        # model = UNet()
        model.load_state_dict(torch.load(self.weight_path))
        model.cuda()
        model.eval()
        for i, path in enumerate(img_pathes):
            basename = os.path.basename(path)
            print('正在预测:%s, 已完成:(%d/%d)' % (basename, i, len(img_pathes)))
            img = cv.imread(path, cv.IMREAD_COLOR)
            pred = predictWithOverlapB(model, img, patch_size=1024)
            cv.imwrite(os.path.join(self.save_path, basename), pred)
        print('预测完毕!')

    def predict_show(self):
        '''
        :return:
        '''
        imgpath = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train-test-show-data\test-new-data\img'
        tif_name = '3439477_01.tif'
        img_path = os.path.join(imgpath, tif_name)
        model = MECNet(visualization=True)
        model.load_state_dict(torch.load(self.weight_path))
        model.cuda()
        model.eval()
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        labelpath = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train-test-show-data\test-new-data\label'
        label = cv.imread(os.path.join(labelpath, tif_name), cv.IMREAD_GRAYSCALE)
        load_data = LoadTest()
        tmp_img = load_data(img)
        with torch.no_grad():
            tmp_img = Variable(tmp_img)
            tmp_img = tmp_img.cuda().unsqueeze(0)
            result, x_visualize = model(tmp_img)
        output = result if not isinstance(result, (list, tuple)) else result[0]
        output = F.sigmoid(output)
        pred = output.data.cpu().numpy().squeeze(0).squeeze(0)  # [0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        pred = pred.astype(np.uint8)

        x_visualize = x_visualize.data.cpu().numpy()  # 用Numpy处理返回的[1,c,m,n]特征图
        x_visualize = np.max(x_visualize, axis=1).reshape(1024, 1024)  # shape为[m,n]，二维
        x_visualize = (
                    ((x_visualize - np.min(x_visualize)) / (np.max(x_visualize) - np.min(x_visualize))) * 255).astype(
            np.uint8)  # 归一化并映射到0-255的整数，方便伪彩色化
        x_visualize = 255 - x_visualize
        x_visualize = cv.applyColorMap(x_visualize, cv.COLORMAP_JET)  # COLORMAP_JET
        plt.subplot(221)
        plt.imshow(img[:, :, ::-1])
        plt.subplot(222)
        label = cv.applyColorMap(255-label, cv.COLORMAP_JET)
        plt.imshow(label)
        plt.subplot(223)
        plt.imshow(pred)
        plt.subplot(224)
        plt.imshow(x_visualize)
        plt.show()

    def predict_feature_map(self):
        '''
        :return:
        '''
        img_pathes = glob(self.imgpath + '/*.tif')
        model = FMNet(visualization=True)
        model.load_state_dict(torch.load(self.weight_path))
        model.cuda()
        model.eval()
        load_data = LoadTest()
        for i, path in enumerate(img_pathes):
            basename = os.path.basename(path)
            print('正在预测:%s, 已完成:(%d/%d)' % (basename, i, len(img_pathes)))
            img = cv.imread(path, cv.IMREAD_COLOR)
            tmp_img = load_data(img)
            with torch.no_grad():
                tmp_img = Variable(tmp_img)
                tmp_img = tmp_img.cuda().unsqueeze(0)
                result, x_visualize = model(tmp_img)
            x_visualize = x_visualize.data.cpu().numpy()  # 得到[1,c,m,n]特征图
            x_visualize = np.max(x_visualize, axis=1).reshape(1024, 1024)  # shape为二维[m,n]
            x_visualize = (
                        ((x_visualize - np.min(x_visualize)) / (np.max(x_visualize) - np.min(x_visualize))) * 255).astype(
                np.uint8)  # 归一化并映射到0-255的整数，方便伪彩色化
            x_visualize = cv.applyColorMap(x_visualize, cv.COLORMAP_JET)  # COLORMAP_JET
            cv.imwrite(os.path.join(self.save_path, os.path.basename(path)), x_visualize)


if __name__ == '__main__':
    # root = r'J:\datasets\water_paper_allmodels_results\aerial_dataset\deeplabv3plus'
    # save_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\clip_data\train_data\test-show-data\deeplabv3plus'

    # root = r'J:\datasets\water_paper_allmodels_results\aerial_dataset\DANet_result'
    # root = r'J:\datasets\water_paper_allmodels_results\aerial_dataset\MGMNet_result'
    root = r'J:\datasets\water_paper_allmodels_results\aerial_dataset\FCN'
    save_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\ablation_feat_map_show'
    model_name = 'FCN'
    save_path = os.path.join(save_path, model_name)
    os.makedirs(save_path, exist_ok=True)
    img_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\ablation_feat_map_show\img'
    # img_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train-test-show-data\test-new-data\img\show-img'

    # weight_path = root + '/Epoch_16_TrainLoss_0.0454_miou_0.9882.pkl'
    weight_path = root + '/Epoch_31_TrainLoss_0.0331_miou_0.9874.pkl'
    predict_fuc = Test(save_path, img_path, weight_path)
    predict_fuc.predict()
    # predict_fuc.predict_show()
    # predict_fuc.predict_feature_map()
