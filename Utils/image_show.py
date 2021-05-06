import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import colors

predict_path = r'E:\zl_datas\paper_experimental_dataset_and_result\aerial_dataset\show_img_test'
save_path = os.path.join(predict_path, 'heat_map')
os.makedirs(save_path, exist_ok=True)


pred_path = os.path.join(predict_path, 'prob', '3447493_53.tif')
img_path = os.path.join(predict_path, 'img', '3447493_53.tif')
label_path = os.path.join(predict_path, 'label', '3447493_53.tif')
ori_img = cv.imread(img_path, cv.IMREAD_COLOR)
prob = cv.imread(pred_path, cv.IMREAD_GRAYSCALE)
label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)


'''方式一'''
# heat_img = cv.applyColorMap(prob, cv.COLORMAP_JET)  #COLORMAP_JET
# heat_label = cv.applyColorMap(label, cv.COLORMAP_JET)  #COLORMAP_JET
# # heat_img = cv.cvtColor(heat_img, cv.COLOR_BGR2RGB)
# # img_add = cv.addWeighted(ori_img, 0.3, heat_img, 0.7, 0)
# cv.imshow('heat_img', heat_img)
# cv.imshow('label', heat_label)
# cv.imshow('img', ori_img)
# cv.waitKey(0)
'''方式二'''

# # aa = plt.imshow(prob, plt.get_cmap('jet'))
# # plt.axis('off')
# # # plt.colorbar(aa) # 添加一个colorbar
# # plt.show()
# fig = plt.figure(1)
# imgs = [prob, prob, prob, prob, prob]
# for i in range(4):
#     plt.subplot(1, 5, i+1)
#     aa = plt.imshow(imgs[i], plt.get_cmap('jet'))
#     plt.axis('off')
#
#
# # plt.subplot(1, 5, 5)
# # plt.tight_layout()
# # plt.axis('off')
# fig.colorbar(aa)
# # plt.colorbar(aa)
# # 去除图像边缘空白
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.margins(0,0)
# plt.show()
# aa = plt.imshow(prob, plt.get_cmap('jet'))
# plt.axis('off')
# plt.colorbar(aa)
# plt.show()
#
#
# # fig.savefig(out_png_path, format='png', transparent=True, dpi=300, pad_inches = 0)

'''方式三'''
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:31:07 2019

@author: BAI Depei
"""


rows, cols = 8, 7
fig, axs = plt.subplots(nrows=rows, ncols=cols,figsize=(16, 16))  # figsize = (width，hight)  3:2
vmin, vmax = 0, 255
for i in range(rows):
    for j in range(cols):
        if i == rows-1 and j == cols-1:
            sc = axs[i, j].imshow(prob, vmin=vmin, vmax=vmax, cmap='jet')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            continue
        axs[i, j].imshow(prob, vmin=vmin, vmax=vmax, cmap='jet')
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

#前面三个子图的总宽度 为 全部宽度的 0.9；剩下的0.1用来放置colorbar
# fig.subplots_adjust(right=0.9)
# #colorbar 左 下 宽 高
# l = 0.92
# b = 0.53
# w = 0.015
# h = 0.25
# #对应 l,b,w,h；设置colorbar位置；
# rect = [l,b,w,h]
# cbar_ax = fig.add_axes(rect)
# # plt.colorbar(sc, cax=cbar_ax)
# plt.colorbar(sc, cax=cbar_ax)
fig.subplots_adjust(right=0.82)
#在原fig上添加一个子图句柄为cbar_ax, 设置其位置为[0.85,0.15,0.05,0.7]
#colorbar 左 下 宽 高
l = 0.85
b = 0.12
w = 0.015
h = 1 - 2*b
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h]
cbar_ax = fig.add_axes(rect)

cb = fig.colorbar(sc, cax = cbar_ax)
plt.savefig('correct_pictures.png',dpi=300,bbox_inches='tight')