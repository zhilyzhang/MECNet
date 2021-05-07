import cv2 as cv
import numpy as np
import os
import scipy.misc

N_row = 6
N_col = 6

patch_size = 1024 + 4
jg = 20

rows = N_row*patch_size + (N_row-1)*jg
cols = N_col*patch_size + (N_col-1)*jg
show_image = np.zeros((rows, cols, 3), dtype=np.uint8)
show_image = show_image + 255
# show_image[:, :, 1] = 255  # 绿色填充
# tif_names = os.listdir(r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\paper-show-data\aerial-images')
# tif_names = os.listdir(r'E:\zl_datas\paper_experimental_dataset_and_result\satelite_dataset\satelite_visualization\img')


def is_tif(tif_name):
    return tif_name.endswith('.tif')


# tif_names = list(filter(is_tif, tif_names))
# tif_names = ['3435483_00.tif', '3435483_70.tif', '3435486_40.tif',
#              '3439477_01.tif', '3439486_52.tif']
#
# tif_names = ['3436486_75.tif', '3435483_70.tif', '3439477_01.tif', '3439493_66.tif', '3439494_29.tif', '3439493_25.tif',
#              '3440488_28_1.tif', '3439486_52.tif', '3447493_0.tif']


tif_names = ['3436486_68.tif', '3435483_72.tif', '3439477_01.tif',  '3435486_40.tif',
             '3447493_48.tif', '3447493_50.tif']

# dir_name = ['aerial-images', 'aerial-labels', 'UNet_result/predict', 'RefineNet_result/predict',
#             'deeplabv3plus_result/predict', 'DANet_result/predict',
#             'CascadePSPNet_result/predict', 'predict-MFCNet/predict']

# dir_name = ['img/predict', 'label', 'FCN/predict', 'FCN_MFC/predict',
#             'MFC_MSP/predict', 'MFCNet/predict']
# dir_name = ['test-new-data/show-img', 'test-new-data/label', 'UNet/predict', 'RefineNet/predict',
#             'deeplabv3plus/predict', 'DANet/predict',
#             'CascadePSP/predict', 'MFCNet/predict']

# dir_name = ['img', 'hot_map', 'FCN', 'FMNet',
#             'MMNet', 'MECNet']

dir_name = ['img', 'label', 'FCN', 'FMNet',
            'MMNet', 'MECNet']
# root = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\paper-show-data'
# root = r'E:\zl_datas\paper_experimental_dataset_and_result\satelite_dataset\satelite_visualization'
# root = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train-test-show-data'
root = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\ablation_feat_map_show'
for i in range(N_row):
    tif_name = tif_names[i]
    for j in range(N_col):
        img = cv.imread(os.path.join(root, dir_name[j], tif_name), cv.IMREAD_COLOR)
        print(os.path.join(root, dir_name[j], tif_name))
        m, n, _ = img.shape
        if n == 512:
            if j == 0:
                img = scipy.misc.imresize(img, (1024, 1024), 'cubic')
            else:
                img = scipy.misc.imresize(img, (1024, 1024), 'nearest')
        tmp = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        # RGB:30 144 255 边框
        tmp[:, :, 0] = 255
        tmp[:, :, 1] = 144
        tmp[:, :, 2] = 30
        tmp[2:-2, 2:-2, :] = img
        show_image[i*(patch_size+jg):i*(patch_size+jg)+patch_size, j*(patch_size+jg):j*(patch_size+jg)+patch_size, :] = tmp
cv.imwrite(os.path.join(root, 'show_img-test-paper.tif'), show_image)