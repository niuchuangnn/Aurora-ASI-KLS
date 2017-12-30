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
    heatmap_folder = '/home/amax/NiuChuang/CAM/aurora_heatmaps/'
    predict_file = '/home/amax/NiuChuang/CAM/aurora_predict.txt'
    type_num = 4

    [names, predicts] = plf.parseNL(predict_file)

    k = 60
    minSize = 50
    sigma = 0.5

    imSize = 440
    eraseMap = np.zeros((imSize, imSize))
    radius = imSize / 2
    centers = np.array([219.5, 219.5])
    for i in range(imSize):
        for j in range(imSize):
            if np.linalg.norm(np.array([i, j]) - centers) > radius + 5:
                eraseMap[i, j] = 1

    cls_confusion = np.zeros((type_num, type_num))
    IoUs = np.zeros(cls_confusion.shape)
    result_file = '../../Data/Results/segmentation/CAM/vgg16_th0.5.txt'
    f = open(result_file, 'w')
    is_detection = False

    for t in range(0, type_num):
        mask_folder_t = data_folder + str(t+1) + '_mask/'
        mask_names = os.listdir(mask_folder_t)
        img_folder_t = data_folder + str(t+1) + '_selected/'
        for i in range(len(mask_names)):
            mask_name = mask_names[i]
            mask_path = mask_folder_t + mask_name
            im_name = mask_name[:-4]

            # for showing individual image
            im_name = 'N20041124G093900'
            t = 2
            mask_folder_t = data_folder + str(t + 1) + '_mask/'
            img_folder_t = data_folder + str(t + 1) + '_selected/'
            mask_path = mask_folder_t + im_name + '.mat'

            predict_label = int(predicts[names.index(im_name)])
            predict_heatmap_path = heatmap_folder + im_name + '_' + str(predict_label) + '.mat'
            predict_heatmap = sio.loadmat(predict_heatmap_path)['heat_map_norm']
            label_mask = load_mask_mat(mask_path)
            img_path = img_folder_t + im_name + '.jpg'
            predict_heatmap = imresize(predict_heatmap, [imSize, imSize]).astype(np.float) / 255.0
            predict_heatmap_th = np.copy(predict_heatmap)
            predict_heatmap_th[np.where(predict_heatmap_th < 0.5)] = 0

            if is_detection:
                th = calculateThreshold(img_path)
                detect_region, F0, remove_labels = detect_regions(img_path, eraseMap, k, minSize, sigma, th)

                predict_segment_labels = set(F0[np.where(predict_heatmap_th>0)])
                predict_mask = np.zeros(F0.shape).astype(np.uint8)
                for sl in predict_segment_labels:
                    if sl not in remove_labels:
                        predict_mask[np.where(F0==sl)] = 1
            else:
                predict_mask = predict_heatmap_th

            cls_confusion[t, predict_label-1] += 1

            intersectionPixelNum = len(np.argwhere((predict_mask * label_mask) > 0))
            unionPixelNum = len(np.argwhere((predict_mask + label_mask) > 0))
            IoU = float(intersectionPixelNum) / float(unionPixelNum)

            IoUs[t, predict_label-1] += IoU

            f.write(im_name + ' ' + str(t) + ' ' + str(predict_label-1) + ' ' + str(IoU) + '\n')
            print im_name + ' ' + str(t) + ' ' + str(predict_label-1) + ' ' + str(IoU)
            if True:
                plt.figure(1)
                plt.imshow(label_mask*255, cmap='gray')
                plt.figure(2)
                plt.imshow(predict_heatmap, cmap='gray')

                plt.figure(3)
                plt.imshow(predict_heatmap_th, cmap='gray')

                predict_heatmap_th[np.where(predict_heatmap_th > 0)] = 255
                imsave('predict_mask_CAM_nodetection.jpg', predict_heatmap_th)

                if is_detection:
                    plt.figure(4)
                    plt.imshow(detect_region*255, cmap='gray')
                plt.figure(5)
                plt.imshow(imread(img_path), cmap='gray')

                plt.figure(6)
                plt.imshow(predict_mask*255, cmap='gray')
                plt.title('predict label: ' + str(predict_label))

                imsave('predict_mask_CAM_detection.jpg', predict_mask * 255)

                plt.show()



                pass
    print cls_confusion
    print IoUs
    rightNums = [cls_confusion[k, k] for k in xrange(type_num)]
    rightNums = np.array(rightNums, dtype='f')
    rightIoUs = [IoUs[k, k] for k in xrange(type_num)]
    print rightIoUs / rightNums

