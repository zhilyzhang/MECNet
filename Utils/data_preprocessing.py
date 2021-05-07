import os
import cv2 as cv
from glob import glob
from shutil import move, copyfile
from tqdm import tqdm
from scipy import misc

'''
去除边缘像素3个
'''
# img_path = r'E:\zl_datas\ZheJiang_data20200803\ChangXing_house_change_datasets_p\building_dataset_p\val_dataset\img_clip'
# label_path = r'E:\zl_datas\ZheJiang_data20200803\ChangXing_house_change_datasets_p\building_dataset_p\val_dataset\label'
#
# save_path = r'E:\zl_datas\ZheJiang_data20200803\ChangXing_house_change_datasets_p\building_dataset_pro\val_dataset'
# save_img = os.path.join(save_path, 'img')
# save_label = os.path.join(save_path, 'label')
#
# img_path_all = glob(os.path.join(img_path, '*.tif'))
# for i, pth in enumerate(img_path_all):
#     basename = os.path.basename(pth)
#     print('doing: %s, %d/%d' % (basename, i, len(img_path_all)))
#     img = cv.imread(pth, cv.IMREAD_COLOR)
#     cv.imwrite(os.path.join(save_img, basename), img[3:-3, 3:-3, :])
#     label = cv.imread(os.path.join(label_path, basename), cv.IMREAD_GRAYSCALE)
#     cv.imwrite(os.path.join(save_label, basename), label[3:-3, 3:-3])

'''
划分训练-测试集
'''
# data_source = r'E:\zl_datas\ZheJiang_data20200803\water_dataset_processed\aerial_dataset\train-data'
# source_img = os.path.join(data_source, 'img')
# source_label = os.path.join(data_source, 'label')
# data_save = r'E:\zl_datas\ZheJiang_data20200803\water_dataset_processed\aerial_dataset\test-data'
# save_img = os.path.join(data_save, 'img')
# save_lab = os.path.join(data_save, 'label')
# os.makedirs(save_img, exist_ok=True)
# os.makedirs(save_lab, exist_ok=True)
# txt = r'E:\zl_datas\ZheJiang_data20200803\aerial_water_dataset_copy\train_dataset_p\train-test.txt'
# key_name = []
# with open(txt, 'r') as f:
#     lines = f.readlines()
#     for per in lines:
#         key_name.append(per.strip())
#
# for tif_name in key_name:
#     basename = tif_name + '.tif'
#     move(os.path.join(source_img, basename), os.path.join(save_img, basename))
#     move(os.path.join(source_label, basename), os.path.join(save_lab, basename))

'''
带重叠的裁剪影像，大小2048，重叠像素256（重叠率1/8）
'''


def clip_image_label(img, label, patch_size, save_path, tif_name, overlap_rate=1/8):
    overlap_len = int(patch_size * overlap_rate)
    stride_len = patch_size - overlap_len
    m, n, _ = img.shape
    tmp_val = (m- overlap_len) // stride_len
    num_m = tmp_val if (m - overlap_len) % stride_len == 0 else tmp_val + 1

    tmp_val = (n - overlap_len) // stride_len
    num_n = tmp_val if (n - overlap_len) % stride_len == 0 else tmp_val + 1
    num = 0
    for i in range(num_m):
        for j in range(num_n):
            if i == num_m - 1 and j != num_n - 1:
                tmp_img = img[-patch_size:, j * stride_len:j * stride_len + patch_size, :]
                tmp_label = label[-patch_size:, j * stride_len:j * stride_len + patch_size]
            elif i != num_m - 1 and j == num_n - 1:
                tmp_img = img[i * stride_len:i * stride_len + patch_size, -patch_size:, :]
                tmp_label = label[i * stride_len:i * stride_len + patch_size, -patch_size:]
            elif i == num_m - 1 and j == num_n - 1:
                tmp_img = img[-patch_size:, -patch_size:, :]
                tmp_label = label[-patch_size:, -patch_size:]
            else:
                tmp_img = img[i * stride_len:i * stride_len + patch_size,
                          j * stride_len:j * stride_len + patch_size, :]
                tmp_label = label[i * stride_len:i * stride_len + patch_size,
                            j * stride_len:j * stride_len + patch_size]
            cv.imwrite(os.path.join(save_path, 'img', tif_name+'_'+str(num)+'.tif'), tmp_img)
            cv.imwrite(os.path.join(save_path, 'label', tif_name+'_'+str(num)+'.tif'), tmp_label)
            num += 1


def clip_data(data_path, save_path, patch_size, overlap_rate=1/8):
    os.makedirs(os.path.join(save_path, 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'label'), exist_ok=True)

    img_path_all = glob(os.path.join(data_path, 'img', '*.tif'))
    for pth in tqdm(img_path_all):
        basename = os.path.basename(pth)
        tif_name = basename.split('.')[0]
        label_path = os.path.join(data_path, 'label', basename)
        img = cv.imread(pth, cv.IMREAD_COLOR)
        label = cv.imread(label_path, cv.IMREAD_COLOR)
        clip_image_label(img, label, patch_size, save_path, tif_name, overlap_rate=overlap_rate)


def label_1to255(label_path):
    label_pathes = glob(os.path.join(label_path, '*.tif'))
    for pth in tqdm(label_pathes):
        label = cv.imread(pth, cv.IMREAD_GRAYSCALE)
        label = label * 255
        cv.imwrite(pth, label)


def rename_dataset(root, fore_name=''):
    img_pathes = glob(os.path.join(root, 'img', '*.tif'))
    for pth in tqdm(img_pathes):
        basename = os.path.basename(pth)
        os.rename(pth, os.path.join(root, 'img', fore_name+'_'+basename))

    label_pathes = glob(os.path.join(root, 'label', '*.tif'))
    for pth in tqdm(label_pathes):
        basename = os.path.basename(pth)
        os.rename(pth, os.path.join(root, 'label', fore_name + '_' + basename))


def rename_(root):
    files = glob(os.path.join(root, '*.*'))
    for pth in tqdm(files):
        basename = os.path.basename(pth)
        newname = basename.replace('label', 'zl')
        os.rename(pth, os.path.join(root, newname))

def check_shape(root):
    imgs = glob(os.path.join(root, 'imgs', '*.tif'))
    for pth in imgs:
        basename = os.path.basename(pth)
        img = cv.imread(pth, cv.IMREAD_COLOR)
        label = cv.imread(os.path.join(root, 'labels', basename), cv.IMREAD_GRAYSCALE)
        ih, iw, _ = img.shape
        lh, lw = label.shape

        if ih != lh or iw!=lw:
            print(basename)
            h = min(ih, lh)
            w = min(iw, lw)
            cv.imwrite(os.path.join(root, 'imgs', basename), img[:h, :w, :])
            cv.imwrite(os.path.join(root, 'labels', basename), label[:h, :w])
        else:
            print(basename, img.shape, label.shape)


if __name__ == '__main__':
    ''''''
    # import torch
    # weight = torch.load('zj_dlinknet.th')
    # torch.save(weight, 'road_weigh.pkl')
#     # data_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train_data'
#     # save_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\clip_data\train_data'
#
    # data_path = r'E:\zl_datas\ZheJiang_data20200803\satellite_water_body\test'
    # # data_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train_data'
    # save_path = r'E:\zl_datas\paper_experimental_dataset_and_result\satelite_dataset\test_data'
    # # save_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\clip_data\train_data'
    # clip_data(data_path, save_path, patch_size=1024, overlap_rate=0.5)
    '''1-255'''
    # label_path = r'F:\dataset\building_dataset\train_data_2014\label'
    # label_1to255(label_path)
    '''rename'''
    # root = r'F:\dataset\building_dataset\s4_total18_data'
    # fore_name = 's4'
    # rename_dataset(root, fore_name=fore_name)
    ''''''
    # root = r'F:\dataset\building_dataset'
    # check_shape(root)
    ''''''
    # label = cv.imread(r'F:\software_save_files\val_data\label\3_0.tif', cv.IMREAD_GRAYSCALE)
    # label = label // 255
    # cv.imwrite(r'F:\software_save_files\val_data\label\3_0.tif', label)
    data_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\paper-show-data\img-paper-detail'
    save_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\paper-show-data\img-paper-detail\clip-512'
    clip_data(data_path, save_path, patch_size=512, overlap_rate=0)

    # data_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\test-data\img\show_papers\img'
    # filenames = os.listdir(data_path)
    # file_name_list = []
    # for file in filenames:
    #     if file.endswith('.tif'):
    #         file_name_list.append(file)
    #
    # tif_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\test-data\label'
    # save_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\test-data\img\show_papers\label'
    # tifs = os.listdir(tif_path)
    # for tif_name in tifs:
    #     if tif_name in file_name_list:
    #         copyfile(os.path.join(tif_path, tif_name), os.path.join(save_path, tif_name))
#
#     '''测试下采样和上采样精度'''
#     scale = 8
#     root = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\val_data\label'
#     save_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\val_data'
#     save_path = os.path.join(save_path, 'label'+str(scale))
#     os.makedirs(save_path, exist_ok=True)
#     tifs = glob(os.path.join(root, '*.tif'))
#     for tif_path in tifs:
#         label = cv.imread(tif_path, cv.IMREAD_GRAYSCALE)
#         m, n = label.shape
#         label = misc.imresize(label, (m//scale, n//scale), interp='nearest')
#         label = misc.imresize(label, (m, n), interp='nearest')
#         cv.imwrite(os.path.join(save_path, os.path.basename(tif_path)), label)
