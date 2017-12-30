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

def load_mask_mat(filePath, var='mask'):
    mask = sio.loadmat(filePath)[var]
    for x in xrange(mask.shape[0]):
        mask[x, :] = mask[x, ::-1]
    for y in xrange(mask.shape[1]):
        mask[:, y] = mask[::-1, y]
    mask = mask.T
    return mask

def SCKLS(paras):
    classHeatMap = region_class_heatMap(paras)
    labels = mapsToLabels(classHeatMap)

    paras['specialType'] = labels  # 0: arc, 1: drapery, 2: radial, 3: hot-spot
    maps3, common_labels, F0 = region_special_map(paras, isReturnMaps=True)

    kls, categoryMap, classMap = mergePatchAndRegion(classHeatMap, maps3, labels, 0.5)
    return labels, kls, categoryMap, classMap, classHeatMap

def testSegmentation(feaType, wordsNum, patchSize, mk, paras, labeledDataFolder, resultSaveFolder, classNum=4,
                     wordsFolder='../../Data/Features/', mini=False, detection=True, merge=True, nk=None):
    paras['sizeRange'] = (patchSize, patchSize)
    paras['patchSize'] = np.array([patchSize, patchSize])
    paras['feaType'] = feaType

    if mini is True:
        mini_str = '_mini'
    else:
        mini_str = ''

    if detection is True:
        detection_str = ''
    else:
        detection_str = '_noDetection'

    if merge is True:
        merge_str = ''
    else:
        merge_str = '_noMerge'

    if nk is not None:
        nk_str = 'nk_'

    if feaType == 'LBP':
        paras['lbp_wordsFile_s1'] = wordsFolder + 'type4_LBPWords_s1_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['lbp_wordsFile_s2'] = wordsFolder + 'type4_LBPWords_s2_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['lbp_wordsFile_s3'] = wordsFolder + 'type4_LBPWords_s3_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['lbp_wordsFile_s4'] = wordsFolder + 'type4_LBPWords_s4_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
    if feaType == 'SIFT':
        paras['sift_wordsFile_s1'] = wordsFolder + 'type4_SIFTWords_s1_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['sift_wordsFile_s2'] = wordsFolder + 'type4_SIFTWords_s2_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['sift_wordsFile_s3'] = wordsFolder + 'type4_SIFTWords_s3_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['sift_wordsFile_s4'] = wordsFolder + 'type4_SIFTWords_s4_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
    if feaType == 'His':
        paras['his_wordsFile_s1'] = wordsFolder + 'type4_HisWords_s1_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['his_wordsFile_s2'] = wordsFolder + 'type4_HisWords_s2_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['his_wordsFile_s3'] = wordsFolder + 'type4_HisWords_s3_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['his_wordsFile_s4'] = wordsFolder + 'type4_HisWords_s4_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
    resultFile = 'segmentation_' + feaType + '_w' + str(wordsNum) + '_s' + str(patchSize) + '_mk' + str(mk) + detection_str + merge_str + nk_str + str(nk) + '.txt'
    f_result = open(resultSaveFolder + resultFile, 'w')
    if mk == 0:
        paras['mk'] = None
    else:
        paras['mk'] = mk
    IoU_accuracy = np.zeros((classNum, ))
    confusionArray_c = np.zeros((classNum, classNum))
    for c in xrange(0, classNum):
        labelImgFolder_c = labeledDataFolder + str(c + 1) + '_selected/'
        labelMaskFolder_c = labeledDataFolder + str(c + 1) + '_mask/'
        imgFiles = os.listdir(labelImgFolder_c)
        for imgName in imgFiles:
            imgFile = labelImgFolder_c + imgName
            imName = imgName[:-4]

            # for showing individual image
            imName = 'N20061114G132700'
            c = 0
            labelImgFolder_c = labeledDataFolder + str(c + 1) + '_selected/'
            labelMaskFolder_c = labeledDataFolder + str(c + 1) + '_mask/'
            imgFile = labelImgFolder_c + imName + '.jpg'

            mask_c = load_mask_mat(labelMaskFolder_c + imName + '.mat')

            if detection is True:
                paras['thresh'] = calculateThreshold(imgFile)
            else:
                paras['thresh'] = 0
            print paras['thresh']
            paras['imgFile'] = imgFile
            im = skimage.io.imread(imgFile)
            if len(im.shape) == 2:
                img = skimage.color.gray2rgb(im)
            paras['img'] = img
            paras['im'] = im
            # imName = imgFile[-20:-4]
            # ----no rotation----
            class_names = ['background', 'arc', 'drapery', 'radial', 'hot-spot']
            labels, kls, categoryMap, classMap, classHeatMap = SCKLS(paras)
            confusionArray_c[c, labels] += 1

            if merge is False:
                kls = np.zeros(kls.shape)
                kls[np.where(categoryMap > 0.3)] = 1
            intersectionPixelNum = len(np.argwhere((kls * mask_c) > 0))
            unionPixelNum = len(np.argwhere((kls + mask_c) > 0))
            IoU = float(intersectionPixelNum) / float(unionPixelNum)
            print 'IoU:', IoU
            if labels == c:
                IoU_accuracy[c] += IoU
            f_result.write(imgName + ' ' + str(c) + ' ' + str(labels) + ' ' + str(IoU) + '\n')

            if True:  # show segmentation results
                plt.figure(10)
                plt.imshow(kls, cmap='gray')
                plt.title(class_names[labels + 1] + '_predict')
                plt.axis('off')

                imsave('predict_mask_proposed.jpg', kls)

                plt.figure(11)
                plt.imshow(mask_c, cmap='gray')
                plt.title(class_names[c + 1] + '_groundTruth')
                plt.axis('off')

                plt.figure(12)
                plt.imshow(im, cmap='gray')
                plt.title('raw image')
                plt.axis('off')
                plt.show()
                # mask_c = mask_c.astype(np.int)
                # kls = kls.astype(np.int)

    f_result.close()
    print confusionArray_c
    accuracy = confusionArray_c / np.sum(confusionArray_c, axis=1).reshape(classNum, 1)
    rightNums = [confusionArray_c[k, k] for k in xrange(classNum)]
    rightNums = np.array(rightNums, dtype='f')
    IoUs = IoU_accuracy / rightNums
    print accuracy
    print rightNums
    print IoUs
    return 0

if __name__ == '__main__':
    paras = {}
    paras['color_space'] = ['rgb']
    paras['ks'] = [30, 50, 100, 150, 200, 250, 300]
    paras['feature_masks'] = [1, 1, 1, 1]
    paras['overlapThresh'] = 0.9
    paras['scoreThresh'] = 0.7
    eraseMapPath = '../../Data/eraseMap.bmp'
    regionModelWeights = '../../Data/region_classification/output/vgg_cnn_m_1024_fast_rcnn_b500_iter_10000.caffemodel'
    regionModelPrototxt = '../../fast-rcnn/models/VGG_CNN_M_1024/test_kls.prototxt'
    proposal_minSize = 100 * 100
    proposal_maxSize = 440 * 220
    paras['regionSizeRange'] = [proposal_minSize, proposal_maxSize]
    if not os.path.exists(eraseMapPath):
        imSize = 440
        eraseMap = np.zeros((imSize, imSize))
        radius = imSize / 2
        centers = np.array([219.5, 219.5])
        for i in range(440):
            for j in range(440):
                if np.linalg.norm(np.array([i, j]) - centers) > 220 + 5:
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

    paras['region_patch_ratio'] = 0.1
    paras['sigma'] = 0.5
    paras['alpha'] = 0.6
    paras['th'] = 0.15
    paras['types'] = ['arc', 'drapery', 'radial', 'hot_spot']

    paras['imResize'] = (256, 256)
    paras['imgSize'] = (440, 440)
    paras['nk'] = 1
    resolution = 1
    gridSize = np.array([resolution, resolution])
    paras['resolution'] = resolution
    paras['gridSize'] = gridSize
    paras['withIntensity'] = 'False'
    paras['diffResolution'] = 'False'
    paras['isSave'] = False
    paras['is_rotate'] = False
    paras['sdaePara'] = None

    paras['returnRegionLabels'] = [0]  # 0: special, 1: rest, 2: common
    paras['train'] = False
    is_showProposals = paras['is_showProposals'] = False

    # feaTypes = ['LBP', 'SIFT', 'His']
    # wordsNums = [50, 100, 200, 500]
    # patchSizes = [8, 16, 24, 32, 40, 48, 56, 64]
    # mks = [0]
    #
    # labeledDataFolder = '../../Data/segmentation_data_v2/'
    # resultSaveFolder = '../../Data/Results/segmentation/modelFS_segV2_FWP_mk0_1/'
    # for feaType in feaTypes:
    #     for wordsNum in wordsNums:
    #         for patchSize in patchSizes:
    #             for mk in mks:
    #                 # if (feaType=='LBP') and ((wordsNum<500) or ((wordsNum==500) and (patchSize<32))):
    #                 #     print feaType, wordsNum, patchSize
    #                 #     continue
    #                 # else:
    #                 result_file = 'segmentation_' + feaType + '_w' + str(wordsNum) + '_s' + str(patchSize) + '_mk' + str(
    #                     mk) + '.txt'
    #                 if not os.path.exists(resultSaveFolder + result_file):
    #                     testSegmentation(feaType, wordsNum, patchSize, mk, paras, labeledDataFolder, resultSaveFolder)
    feaTypes = ['LBP']
    wordsNums = [100]
    patchSizes = [16]
    # mks = range(0, 7000, 200)
    mks = [100]
    # nk = range(1, 11, 2)
    nk = [1]
    detection = True
    merge = True

    if merge is True:
        merge_str = ''
    else:
        merge_str = '_noMerge'

    if detection is True:
        detection_str = ''
    else:
        detection_str = '_noDetection'

    labeledDataFolder = '../../Data/segmentation_data_v2/'
    resultSaveFolder = '../../Data/Results/segmentation/segV2_LBP_s16_w100_mk0_nk/'
    for feaType in feaTypes:
        for wordsNum in wordsNums:
            for patchSize in patchSizes:
                for mk in mks:
                    for nk_i in nk:
                        paras['nk'] = nk_i
                        result_file = 'segmentation_' + feaType + '_w' + str(wordsNum) + '_s' + str(
                            patchSize) + '_mk' + str(mk) + detection_str + merge_str + 'nk_' + str(nk_i) + '.txt'
                        # if not os.path.exists(resultSaveFolder + result_file):
                        testSegmentation(feaType, wordsNum, patchSize, mk, paras, labeledDataFolder,
                                             resultSaveFolder, detection=detection, merge=merge, nk=nk_i)
