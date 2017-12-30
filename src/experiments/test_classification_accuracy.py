from src.localization.region_category_map import region_special_map, showSelectRegions, showMaps3
from src.localization.regions_classes_map import region_class_heatMap, visRegionClassHeatMap
import sys
sys.path.insert(0, '../../fast-rcnn/caffe-fast-rcnn/python')
import caffe
from scipy.misc import imread, imsave
import skimage.io
import os
import numpy as np
from sklearn.decomposition import PCA
from skimage.transform import rotate
import math
import matplotlib.pyplot as plt
from src.localization.klsSegmentation import mapsToLabels, mergePatchAndRegion
import scipy.io as sio
from src.local_feature.adaptiveThreshold import calculateThreshold
import random
from src.localization.generateSubRegions import detect_regions

def CKLS(paras, detection_mask=None):
    classHeatMap = region_class_heatMap(paras)
    labels = mapsToLabels(classHeatMap, detection_mask=detection_mask)
    return labels

if __name__ == '__main__':
    paras = {}

    paras['color_space'] = ['rgb']
    paras['ks'] = [30, 50, 100, 150, 200, 250, 300]
    paras['feature_masks'] = [1, 1, 1, 1]
    paras['overlapThresh'] = 0.9
    paras['scoreThresh'] = 0.7

    eraseMapPath = '../../Data/eraseMap.bmp'
    regionModelWeights = '../../Data/region_classification/output/vgg_cnn_m_1024_fast_rcnn_F_iter_50000.caffemodel'
    regionModelPrototxt = '../../fast-rcnn/models/VGG_CNN_M_1024/test_kls.prototxt'
    proposal_minSize = 100 * 100
    proposal_maxSize = 440 * 440
    paras['regionSizeRange'] = [proposal_minSize, proposal_maxSize]
    if not os.path.exists(eraseMapPath):
        imSize = 440
        eraseMap = np.zeros((imSize, imSize))
        radius = imSize / 2
        centers = np.array([219.5, 219.5])
        for i in range(440):
            for j in range(440):
                if np.linalg.norm(np.array([i, j]) - centers) > 220:
                    eraseMap[i, j] = 1
        imsave(eraseMapPath, eraseMap)
    else:
        eraseMap = imread(eraseMapPath) / 255
    paras['eraseMap'] = eraseMap

    gpu_id = 0
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(regionModelPrototxt, regionModelWeights, caffe.TEST)
    paras['net'] = net

    paras['k'] = 60
    paras['minSize'] = 50
    paras['patchSize'] = np.array([16, 16])
    paras['region_patch_ratio'] = 0.1
    paras['sigma'] = 0.5
    paras['alpha'] = 0.6
    paras['th'] = 0.15
    paras['types'] = ['arc', 'drapery', 'radial', 'hot_spot']

    paras['sizeRange'] = (16, 16)
    paras['imResize'] = (256, 256)
    paras['imgSize'] = (440, 440)
    paras['nk'] = 1
    resolution = 1
    gridSize = np.array([resolution, resolution])
    paras['resolution'] = resolution
    paras['gridSize'] = gridSize
    # im = np.array(imread(imgFile), dtype='f') / 255
    # paras['im'] = im

    paras['isSave'] = False
    paras['is_rotate'] = False

    paras['returnRegionLabels'] = [0]  # 0: special, 1: rest, 2: common
    paras['train'] = False
    is_showProposals = paras['is_showProposals'] = False

    resultsSaveFolder = '../../Data/Results/classification/'
    result_cls = 'result_classification_test100_440.txt'
    classNum = 4
    confusionArray_c = np.zeros((classNum, classNum))
    IoU_accuracy = np.zeros((classNum, ))
    labelDataFolder = '../../Data/alllbp04-09Pri_70K_selected/'
    # imgFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/N20031222G074652.bmp'

    f_cls = open(resultsSaveFolder+result_cls, 'w')
    test_num_per_class = 500
    for c in xrange(0, classNum):
        labelImgFolder_c = labelDataFolder + str(c+1)
        imgFiles = os.listdir(labelImgFolder_c)
        images_num_c = len(imgFiles)
        random.seed(10)
        random.shuffle(imgFiles)

        for im_idx in range(test_num_per_class):
            imgName = imgFiles[im_idx]
            # imgName = 'N20080109G134100.jpg'
            imgFile = labelImgFolder_c + '/' + imgName
            img_c = imread(imgFile)
            imName = imgName[:-4]
            print imName

            # thresh = calculateThreshold(imgFile)
            # detection_mask, _, _ = detect_regions(imgFile, eraseMap, 60, 50, 0.5, thresh)
            detection_mask = None

            paras['imgFile'] = imgFile
            im = skimage.io.imread(imgFile)
            if len(im.shape) == 2:
                img = skimage.color.gray2rgb(im)
            paras['img'] = img
            paras['im'] = im
            # imName = imgFile[-20:-4]
            # ----no rotation----
            class_names = ['background', 'arc', 'drapery', 'radial', 'hot-spot']
            labels = CKLS(paras, detection_mask=detection_mask)
            confusionArray_c[c, labels] += 1
            f_cls.write(imName + ' ' + str(c) + ' ' + str(labels) + '\n')
    print confusionArray_c
    accuracy = confusionArray_c / np.sum(confusionArray_c, axis=1).reshape(classNum, 1)
    print accuracy
