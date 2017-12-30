import skimage.data
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../../selective_search_py')
import argparse
import warnings
import numpy
import skimage.io
import features
import color_space
# import selective_search
import src.util.paseLabeledFile as plf
import segment
import src.preprocess.esg as esg
from scipy.misc import imsave
from src.local_feature.adaptiveThreshold import calculateThreshold
from scipy.misc import imread, imresize
import os
import scipy.io as sio
from src.experiments.test_segmentation_accuracy import load_mask_mat
from src.localization.generateSubRegions import detect_regions

if __name__ == '__main__':
    data_folder = '/home/amax/NiuChuang/KLSA-auroral-images/Data/segmentation_data_v2/'
    type_num = 4

    k = 100
    minSize = 300
    sigma = 0.5

    imSize = 440
    eraseMap = np.zeros((imSize, imSize))
    radius = imSize / 2
    centers = np.array([219.5, 219.5])
    for i in range(imSize):
        for j in range(imSize):
            if np.linalg.norm(np.array([i, j]) - centers) > radius + 5:
                eraseMap[i, j] = 1

    IoUs = np.zeros((type_num,))
    nums = np.zeros((type_num,))
    result_file = '../../Data/Results/segmentation/RD/rd.txt'
    f = open(result_file, 'w')

    for t in range(0, type_num):
        mask_folder_t = data_folder + str(t+1) + '_mask/'
        mask_names = os.listdir(mask_folder_t)
        img_folder_t = data_folder + str(t+1) + '_selected/'
        for i in range(len(mask_names)):
            mask_name = mask_names[i]
            mask_path = mask_folder_t + mask_name
            im_name = mask_name[:-4]

            label_mask = load_mask_mat(mask_path)
            img_path = img_folder_t + im_name + '.jpg'
            th = calculateThreshold(img_path)
            detect_region, F0, remove_labels = detect_regions(img_path, eraseMap, k, minSize, sigma, th)
            # predict_heatmap = imresize(predict_heatmap, [imSize, imSize]).astype(np.float) / 255.0
            # predict_heatmap_th = np.copy(predict_heatmap)
            # predict_heatmap_th[np.where(predict_heatmap_th < 0.5)] = 0
            # predict_segment_labels = set(F0[np.where(predict_heatmap_th>0)])
            # predict_mask = np.zeros(F0.shape).astype(np.uint8)
            # for sl in predict_segment_labels:
            #     if sl not in remove_labels:
            #         predict_mask[np.where(F0==sl)] = 1
            # cls_confusion[t, predict_label-1] += 1

            intersectionPixelNum = len(np.argwhere((detect_region * label_mask) > 0))
            unionPixelNum = len(np.argwhere((detect_region + label_mask) > 0))
            IoU = float(intersectionPixelNum) / float(unionPixelNum)

            IoUs[t] += IoU
            nums[t] += 1

            f.write(im_name + ' ' + str(t) + ' ' + str(t) + ' ' + str(IoU) + '\n')
            print im_name + ' ' + str(t) + ' ' + str(t) + ' ' + str(IoU)
            if False:
                plt.figure(1)
                plt.imshow(label_mask*255, cmap='gray')
                plt.figure(4)
                plt.imshow(detect_region*255, cmap='gray')
                plt.figure(5)
                plt.imshow(imread(img_path), cmap='gray')

                plt.show()
    print IoUs / nums

