import cv2 as cv
import numpy as np
import os
import argparse


image_format = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff']


def get_files(root):
    def format_filter(filename):
        for im_format in image_format:
            if filename.lower().endswith(im_format):
                # lower() 忽略大小格式
                return True
        return False
    img_files = os.listdir(root)
    new_img_files = list(filter(format_filter, img_files))
    file_pathes = [os.path.join(root, filename) for filename in new_img_files]
    return file_pathes


def check_train_data_coarse(img_path, label_path):
    '''
    :param root: 工程文件路径
    :return:
    '''
    img_pathes = get_files(img_path)
    label_pathes = get_files(label_path)
    if len(img_pathes) != len(label_pathes):
        return 'img与label数量不一致！'
    impt = [os.path.basename(p) for p in img_pathes]
    for lp in label_pathes:
        basename = os.path.basename(lp)
        if basename not in impt:
            return '文件名不统一，如：' + basename
    return '数据量检查、文件名检查无误！'


def check_train_data_detail(img_path, label_path, class_num=1):
    '''
    :param root: 工程文件路径
    :param class_num: 类别数
    :return:
    '''
    img_pathes = get_files(img_path)

    for impath in img_pathes:
        basename = os.path.basename(impath)
        img = cv.imread(impath, cv.IMREAD_COLOR)
        mx, nx, _ = img.shape
        label = cv.imread(os.path.join(label_path, basename), cv.IMREAD_GRAYSCALE)
        my, ny = label.shape
        if mx != my or nx != ny:
            return 'img与label的shape大小不一致，如：' + basename
        value_categories = np.unique(label)
        value_categories[value_categories == 255] = 0
        value_categories = np.unique(value_categories)
        if value_categories.max() > class_num-1:
            return 'label的类别标注值不正确，如：' + basename
    return 'shape与类别检查无误！'


if __name__ == '__main__':
    Aparse = argparse.ArgumentParser(description='to img_path and label_path')
    Aparse.add_argument("-i", "--img_path", help='to attain the shps')
    Aparse.add_argument("-l", "--label_path", help='to attain the raster')
    args = Aparse.parse_args()
    img_path = args.img_path  # to get corresponding image
    label_path = args.label_path
    print('waiting!')
    print(check_train_data_coarse(img_path, label_path))
    print(check_train_data_detail(img_path, label_path, class_num=1))
    print('over!')