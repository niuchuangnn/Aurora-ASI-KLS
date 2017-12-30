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
from src.local_feature.adaptiveThreshold import calculateThreshold
import scipy.io as sio

def angleFromCommon(F0, commonLabels):
    coordinates_special = np.zeros((0, 2))
    for ls in commonLabels:
        # print 'ls: ' + str(ls)
        coordinates_special = np.vstack([coordinates_special, np.argwhere(F0 == ls)])
    # print 'coordinates_special', coordinates_special
    pca = PCA()
    pca.fit(coordinates_special)

    components = pca.components_
    main_ax = components[0]
    angle = math.atan(main_ax[0] / main_ax[1]) * (180.0 / math.pi)
    return angle

def mapsToLabels(classHeatMaps, detection_mask=None):
    regionHeatSizes = np.zeros((classHeatMaps.shape[2]-1, ))
    for i in xrange(1, classHeatMaps.shape[2]):
        map_i = classHeatMaps[:, :, i]
        if detection_mask is not None:
            map_i *= detection_mask
        regionHeatSizes[i-1] = len(list(map_i[np.where(map_i != 0)].flatten()))
    label = regionHeatSizes.argmax()
    # maxSize = regionHeatSizes[label]
    # labels = []
    # for j in xrange(len(regionHeatSizes)):
    #     if regionHeatSizes[j] / maxSize > 0.85:
    #         labels.append(j)
    return label

def load_mask_mat(filePath, var='mask'):
    mask = sio.loadmat(filePath)[var]
    for x in xrange(mask.shape[0]):
        mask[x, :] = mask[x, ::-1]
    for y in xrange(mask.shape[1]):
        mask[:, y] = mask[::-1, y]
    mask = mask.T
    return mask

def mergePatchAndRegion(classHeatMaps, categoryHeatMaps, labels, th):
    classHeatMap = classHeatMaps[:, :, labels+1]
    categoryHeatMap = categoryHeatMaps.values()[0][0]
    mergeRusults = (classHeatMap + categoryHeatMap) / 2
    # mergeRusults = classHeatMap * categoryHeatMap
    mergeRusults[np.where(mergeRusults > th)] = 1
    mergeRusults[np.where(mergeRusults <= th)] = 0
    return mergeRusults, categoryHeatMap, classHeatMap

if __name__ == '__main__':
    paras = {}
    im_name = 'N20041122G074401'
    imgFolder = '../../Data/segmentation_data_v2/'
    c = 4
    imgFile = imgFolder + str(c) + '_selected/' + im_name + '.jpg'
    imgMask = imgFolder + str(c) + '_mask/' + im_name + '.mat'
    mask_c = load_mask_mat(imgMask)
    paras['imgFile'] = imgFile
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
    im = skimage.io.imread(imgFile)
    if len(im.shape) == 2:
        img = skimage.color.gray2rgb(im)
    paras['img'] = img
    paras['im'] = im
    gpu_id = 3
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(regionModelPrototxt, regionModelWeights, caffe.TEST)
    paras['net'] = net
    paras['k'] = 60
    paras['minSize'] = 50
    paras['patchSize'] = np.array([28, 28])
    paras['region_patch_ratio'] = 0.1
    paras['sigma'] = 0.5
    paras['alpha'] = 0.6
    paras['th'] = 0.15
    paras['types'] = ['arc', 'drapery', 'radial', 'hot_spot']
    # paras['lbp_wordsFile_s1'] = '../../Data/Features/type4_LBPWords_s1_s16_b300_w500.hdf5'
    # paras['lbp_wordsFile_s2'] = '../../Data/Features/type4_LBPWords_s2_s16_b300_w500.hdf5'
    # paras['lbp_wordsFile_s3'] = '../../Data/Features/type4_LBPWords_s3_s16_b300_w500.hdf5'
    # paras['lbp_wordsFile_s4'] = '../../Data/Features/type4_LBPWords_s4_s16_b300_w500.hdf5'
    # paras['sift_wordsFile_s1'] = '../../Data/Features/type4_SIFTWords_s1_diffResolution_b500_intensity.hdf5'
    # paras['sift_wordsFile_s2'] = '../../Data/Features/type4_SIFTWords_s2_diffResolution_b500_intensity.hdf5'
    # paras['sift_wordsFile_s3'] = '../../Data/Features/type4_SIFTWords_s3_diffResolution_b500_intensity.hdf5'
    # paras['sift_wordsFile_s4'] = '../../Data/Features/type4_SIFTWords_s4_diffResolution_b500_intensity.hdf5'

    paras['imgSize'] = (440, 440)
    paras['nk'] = 1
    resolution = 1
    gridSize = np.array([resolution, resolution])
    paras['resolution'] = resolution
    paras['gridSize'] = gridSize
    # im = np.array(imread(imgFile), dtype='f') / 255
    # paras['im'] = im

    # sdaePara = {}
    # sdaePara['weight_d'] = '../../Data/autoEncoder/layer_diff_mean_s16_final.caffemodel'
    # # sdaePara['weight_s'] = '../../Data/autoEncoder/layer_same_mean_s16_final.caffemodel'
    # sdaePara['weight'] = '../../Data/autoEncoder/layer_same_mean_s28_special_final.caffemodel'
    # sdaePara['net'] = '../../Data/autoEncoder/test_net.prototxt'
    # # sdaePara['meanFile'] = '../../Data/patchData_mean_s16.txt'
    # sdaePara['meanFile'] = '../../Data/patchData_mean_s28_special.txt'
    # sdaePara['patchMean'] = False
    # # layerNeuronNum = [28 * 28, 2000, 1000, 500, 128]
    # layerNeuronNum = [28 * 28, 1000, 1000, 500, 64]
    # sdaePara['layerNeuronNum'] = layerNeuronNum

    paras['sdaePara'] = None

    feaType = 'LBP'
    paras['feaType'] = feaType
    paras['mk'] = 0
    patchSize = 16
    wordsNum = 100
    classNum = 4
    wordsFolder = '../../Data/Features/'
    for c in range(classNum):
        key = feaType.lower() + '_wordsFile_s' + str(c+1)
        paras[key] = wordsFolder + 'type4_' + feaType + 'Words_s' + str(c+1) + '_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'

    paras['sizeRange'] = (patchSize, patchSize)

    paras['withIntensity'] = False
    paras['diffResolution'] = False
    paras['isSave'] = False

    paras['is_rotate'] = False

    paras['returnRegionLabels'] = [0]  # 0: special, 1: rest, 2: common
    paras['train'] = False
    is_showProposals = paras['is_showProposals'] = True

    resultsSaveFolder = '../../Data/Results/segmentation/'
    imName = imgFile[-20:-4]
    rotation = False
    # ----no rotation----
    if rotation == False:
        classHeatMap = region_class_heatMap(paras)
        class_names = ['background', 'arc', 'drapery', 'radial', 'hot-spot']
        labels = mapsToLabels(classHeatMap)
        # visRegionClassHeatMap(classHeatMap, class_names)

        paras['specialType'] = labels  # 0: arc, 1: drapery, 2: radial, 3: hot-spot
        paras['thresh'] = calculateThreshold(imgFile)
        print calculateThreshold(imgFile)
        maps3, common_labels, F0 = region_special_map(paras, isReturnMaps=True)

         # showMaps3(maps3)
        # showSelectRegions(F0, common_labels)
        # kls = mergePatchAndRegion(classHeatMap, maps3, labels, paras['th']*0.95)
        kls, categoryMap, classMap = mergePatchAndRegion(classHeatMap, maps3, labels, 0.5)

        intersectionPixelNum = len(np.argwhere((kls * mask_c) > 0))
        unionPixelNum = len(np.argwhere((kls + mask_c) > 0))
        IoU = float(intersectionPixelNum) / float(unionPixelNum)
        print 'IoU:', IoU

        for ci in xrange(classHeatMap.shape[2]):
            imsave(resultsSaveFolder+imName+'_classMap_'+str(ci)+'.jpg', classHeatMap[:, :, ci])
        imsave(resultsSaveFolder+imName+'_categoryMap.jpg', categoryMap)
        plt.figure(10)
        plt.imshow(kls, cmap='gray')
        class_names = ['background', 'arc', 'drapery', 'radial', 'hot-spot']
        plt.title(class_names[labels+1])
        plt.axis('off')
        imsave(resultsSaveFolder+imName+'_mask.jpg', kls)
        plt.figure(11)
        plt.imshow(im, cmap='gray')
        plt.title('raw image')
        plt.axis('off')
        imsave(resultsSaveFolder + imName + '_raw.jpg', im)

        kls_color = np.zeros(img.shape, dtype='uint8')
        kls_color[:, :, 0][np.where(kls==1)] = 255
        alpha = 0.2
        addImg = (kls_color * alpha + img * (1. - alpha)).astype(np.uint8)
        plt.figure(12)
        plt.imshow(addImg)
        plt.title('add image')
        plt.axis('off')
        imsave(resultsSaveFolder + imName + '_merge.jpg', addImg)
        plt.show()
    else:
        # -----rotate image----
        paras['th'] = 0.5
        paras['specialType'] = 1  # 0: arc, 1: drapery, 2: radial, 3: hot-spot
        maps3, common_labels, F0 = region_special_map(paras, isReturnMaps=True)
        # showSelectRegions(F0, common_labels)
        # showMaps3(maps3)
        angle = angleFromCommon(F0, common_labels)
        paras['th'] = 0.15
        # showSelectRegions(F0, common_labels, angle)
        # F0, region_labels, eraseLabels = region_special_map(paras)
        # showSelectRegions(F0, region_labels)
        # plt.figure(8)
        # plt.imshow(paras['im'], cmap='gray')
        # plt.figure(9)
        # plt.imshow(paras['img'])
        paras['im'] = rotate(im, angle, preserve_range=True).astype(np.uint8)
        paras['img'] = rotate(img, angle, preserve_range=True).astype(np.uint8)
        # plt.figure(6)
        # plt.imshow(paras['im'], cmap='gray')
        # plt.figure(7)
        # plt.imshow(paras['img'])
        # plt.show()
        classHeatMap = region_class_heatMap(paras)
        class_names = ['background', 'arc', 'drapery', 'radial', 'hot-spot']
        # visRegionClassHeatMap(classHeatMap, class_names)
        labels = mapsToLabels(classHeatMap)
        paras['specialType'] = labels  # 0: arc, 1: drapery, 2: radial, 3: hot-spot
        maps3, common_labels, F0 = region_special_map(paras, isReturnMaps=True)
        # showSelectRegions(F0, common_labels)
        # showMaps3(maps3)
        kls = mergePatchAndRegion(classHeatMap, maps3, labels, 0.5)
        plt.figure(10)
        plt.imshow(kls, cmap='gray')
        plt.title(class_names[labels+1] + ' no back rotation')
        kls_rotate = rotate(kls, -angle, preserve_range=True)
        plt.figure(11)
        plt.imshow(kls_rotate, cmap='gray')
        plt.title(class_names[labels + 1] + ' back rotation')
        plt.figure(12)
        plt.imshow(im, cmap='gray')
        plt.title('raw image')
        plt.figure(13)
        plt.imshow(paras['im'], cmap='gray')
        plt.title('rotated image')

        kls_color = np.zeros(img.shape, dtype='uint8')
        kls_color[:, :, 0][np.where(kls_rotate==1)] = 255
        alpha = 0.2
        addImg = (kls_color * alpha + img * (1. - alpha)).astype(np.uint8)
        plt.figure(14)
        plt.imshow(addImg)
        plt.title('add image')
        plt.show()