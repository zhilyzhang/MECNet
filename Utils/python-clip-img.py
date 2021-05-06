import cv2 as cv
import os


tif_name = '3440488_28.tif'

path_list = [r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train-test-show-data\test-new-data\show-img',
             r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train-test-show-data\test-new-data\label',
             r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train-test-show-data\UNet\predict',
             r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train-test-show-data\RefineNet\predict',
             r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train-test-show-data\MFCNet\predict',
             r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train-test-show-data\deeplabv3plus\predict',
             r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train-test-show-data\DANet\predict',
             r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\train-test-show-data\CascadePSP\predict']

patch_size = 512
for i, pth in enumerate(path_list):

    if i == 0:
        img = cv.imread(os.path.join(pth, tif_name), cv.IMREAD_COLOR)
        n = 0
        for x in range(2):
            for y in range(2):
                tm = img[x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size, :]
                cv.imwrite(os.path.join(pth, tif_name[:-4]+'_'+str(n)+'.tif'), tm)
                n += 1
    else:
        im = cv.imread(os.path.join(pth, tif_name), cv.IMREAD_GRAYSCALE)
        n = 0
        for x in range(2):
            for y in range(2):
                tm = im[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size]
                cv.imwrite(os.path.join(pth, tif_name[:-4] + '_' + str(n) + '.tif'), tm)
                n += 1
