import numpy as np
import src.localization.classHeatMap as chm
import src.localization.generateSubRegions as gsr
import src.local_feature.generateLocalFeatures as glf
import src.local_feature.cascadefeas as casf
from scipy.misc import imread, imresize, imsave
import h5py
import matplotlib.pyplot as plt
import os
from skimage.transform import rotate
import skimage
from src.local_feature.adaptiveThreshold import calculateThreshold

def region_category_map(paras):
    imgFile = paras['imgFile']
    sift_wordsFile_h1 = paras['sift_wordsFile_h1']
    sift_wordsFile_h2 = paras['sift_wordsFile_h2']
    sdae_wordsFile_h1_d = paras['sdae_wordsFile_h1_d']
    sdae_wordsFile_h2_d = paras['sdae_wordsFile_h2_d']
    sdae_wordsFile_h1_s = paras['sdae_wordsFile_h1_s']
    sdae_wordsFile_h2_s = paras['sdae_wordsFile_h2_s']
    lbp_wordsFile_h1 = paras['lbp_wordsFile_h1']
    lbp_wordsFile_h2 = paras['lbp_wordsFile_h2']
    cascade_wordsFile = paras['cascade_wordsFile']
    siftFeaFile_reduce = paras['SIFTFeaFile_reduce']
    sdaeFeaFile_reduce_d = paras['SDAEFeaFile_reduce_d']
    sdaeFeaFile_reduce_s = paras['SDAEFeaFile_reduce_s']
    lbpFeaFile_reduce = paras['LBPFeaFile_reduce']
    k = paras['k']
    minSize = paras['minSize']
    patchSize = paras['patchSize']
    region_patch_ratio = paras['region_patch_ratio']
    sigma = paras['sigma']
    th = paras['th']
    sizeRange = paras['sizeRange']
    nk = paras['nk']
    gridSize = paras['gridSize']
    eraseMap = paras['eraseMap']
    im = paras['im']
    sdaePara = paras['sdaePara']
    feaType = paras['feaType']
    if feaType == 'SIFT':
        wordsFile_h1 = sift_wordsFile_h1
        wordsFile_h2 = sift_wordsFile_h2
        feaType_r = feaType
    if feaType == 'LBP':
        wordsFile_h1 = lbp_wordsFile_h1
        wordsFile_h2 = lbp_wordsFile_h2
        feaType_r = feaType
    if feaType == 'SDAE_d':
        wordsFile_h1 = sdae_wordsFile_h1_d
        wordsFile_h2 = sdae_wordsFile_h2_d
        sdaePara['weight'] = sdaePara['weight_d']
        sdaePara['patchMean'] = True
        feaType_r = 'SDAE'
    if feaType == 'SDAE_s':
        wordsFile_h1 = sdae_wordsFile_h1_s
        wordsFile_h2 = sdae_wordsFile_h2_s
        sdaePara['weight'] = sdaePara['weight_s']
        sdaePara['patchMean'] = False
        feaType_r = 'SDAE'
    if feaType == 'cascade':
        sift_f = h5py.File(siftFeaFile_reduce, 'r')
        sdae_f_d = h5py.File(sdaeFeaFile_reduce_d, 'r')
        sdae_f_s = h5py.File(sdaeFeaFile_reduce_s, 'r')
        lbp_f = h5py.File(lbpFeaFile_reduce, 'r')
        sift_u = np.array(sift_f.get('uSet/u'))
        lbp_u = np.array(lbp_f.get('uSet/u'))
        sdae_u_d = np.array(sdae_f_d.get('uSet/u'))
        sdae_u_s = np.array(sdae_f_s.get('uSet/u'))
        feaType_r = feaType

    im_name = imgFile[-20:-4]
    F0, region_patch_list = gsr.generate_subRegions(imgFile, patchSize, region_patch_ratio, eraseMap, k, minSize, sigma)
    maps2by2 = {}
    for ri in range(len(region_patch_list)):
        r = region_patch_list[ri]
        batchSize = len(r)
        inputShape = (batchSize, 1, sizeRange[0], sizeRange[0])
        sdaePara['inputShape'] = inputShape
        if len(r) != 0:
            if feaType_r != 'cascade':
                feaVectors, posVectors = glf.genImgLocalFeas(imgFile, feaType_r, gridSize, sizeRange,
                                                             gridList=r, sdaePara=sdaePara)
                labels = chm.calPatchLabels2by2(wordsFile_h1, wordsFile_h2, feaVectors, nk)
            else:
                feaVectors, posVectors = casf.extractCascadeFeatures(imgFile, sift_u, lbp_u, sdae_u_d, sdae_u_s,
                                                                     r, gridSize, sizeRange, sdaePara)
                labels = chm.calPatchLabels2by2_noH(cascade_wordsFile, feaVectors, nk)

            for k, v in labels.iteritems():
                v = list(v.flatten())
                if k not in maps2by2:
                    maps2by2[k] = np.zeros((3, F0.shape[0], F0.shape[1]))
                c1 = float(v.count(0)) / float(len(v))
                c2 = float(v.count(1)) / float(len(v))
                cc = float(v.count(2)) / float(len(v))
                cs = np.array([c1, c2, cc])
                cs[np.where(cs < th)] = 0

                maps2by2[k][0][np.where(F0 == ri)] = cs[0]
                maps2by2[k][1][np.where(F0 == ri)] = cs[1]
                maps2by2[k][2][np.where(F0 == ri)] = cs[2]

    for c, m in maps2by2.iteritems():
        map3 = np.transpose(m, (1, 0, 2)).reshape(440, 440 * 3)
        map3 = np.append(map3, im, axis=1)
        imsave(im_name + '_' + feaType + '_' + c + '_region' + '.jpg', map3)

def region_special_map(paras, isReturnMaps=None, returnFilteroutLabels=False, no_region=None):
    imgFile = paras['imgFile']
    img = paras['img']
    k = paras['k']
    minSize = paras['minSize']
    patchSize = paras['patchSize']
    region_patch_ratio = paras['region_patch_ratio']
    sigma = paras['sigma']
    th = paras['th']
    sizeRange = paras['sizeRange']
    nk = paras['nk']
    gridSize = paras['gridSize']
    eraseMap = paras['eraseMap']
    im = paras['im']
    feaType = paras['feaType']

    sdaePara = paras['sdaePara']
    types = paras['types']
    withIntensity = paras['withIntensity']
    diffResolution = paras['diffResolution']
    isSave = paras['isSave']
    specialType = paras['specialType']
    returnRegionLabels = paras['returnRegionLabels']
    train = paras['train']
    is_rotate = paras['is_rotate']

    if feaType == 'His':
        his_wordsFile_s1 = paras['his_wordsFile_s1']
        his_wordsFile_s2 = paras['his_wordsFile_s2']
        his_wordsFile_s3 = paras['his_wordsFile_s3']
        his_wordsFile_s4 = paras['his_wordsFile_s4']
        wordsFile_s = [his_wordsFile_s1, his_wordsFile_s2, his_wordsFile_s3, his_wordsFile_s4]

    if feaType == 'LBP':
        lbp_wordsFile_s1 = paras['lbp_wordsFile_s1']
        lbp_wordsFile_s2 = paras['lbp_wordsFile_s2']
        lbp_wordsFile_s3 = paras['lbp_wordsFile_s3']
        lbp_wordsFile_s4 = paras['lbp_wordsFile_s4']
        wordsFile_s = [lbp_wordsFile_s1, lbp_wordsFile_s2, lbp_wordsFile_s3, lbp_wordsFile_s4]

    if feaType == 'SIFT':
        sift_wordsFile_s1 = paras['sift_wordsFile_s1']
        sift_wordsFile_s2 = paras['sift_wordsFile_s2']
        sift_wordsFile_s3 = paras['sift_wordsFile_s3']
        sift_wordsFile_s4 = paras['sift_wordsFile_s4']
        wordsFile_s = [sift_wordsFile_s1, sift_wordsFile_s2, sift_wordsFile_s3, sift_wordsFile_s4]

    if feaType == 'SDAE':
        sdae_wordsFile_s1 = paras['sdae_wordsFile_s1']
        sdae_wordsFile_s2 = paras['sdae_wordsFile_s2']
        sdae_wordsFile_s3 = paras['sdae_wordsFile_s3']
        sdae_wordsFile_s4 = paras['sdae_wordsFile_s4']
        wordsFile_s = [sdae_wordsFile_s1, sdae_wordsFile_s2, sdae_wordsFile_s3, sdae_wordsFile_s4]

    thresh = paras['thresh']
    mk = paras['mk']
    im_name = imgFile[-20:-4]

    if no_region is not None:
        feaVectors, posVectors = glf.genImgLocalFeas(imgFile, feaType, gridSize, sizeRange,
                                                     gridList=None, sdaePara=sdaePara, withIntensity=withIntensity)
        labelVec = chm.calPatchLabels2(wordsFile_s[specialType], feaVectors, k=nk, two_classes=['1', '2'], isH1=True, mk=mk)
        heat_map = chm.generateClassHeatMap(labelVec, posVectors, 3, patchSize, isHierarchy=False, threshold_h1=th, threshold_h2=th)

        for ii in range(3):
            imsave('heatmap_noregion_'+str(ii)+'.jpg', heat_map[ii, :, :])

        map3 = np.transpose(heat_map, (1, 0, 2)).reshape(440, 440 * 3)
        map3 = np.append(im, map3, axis=1)
        if isSave:
            imsave(im_name+'_heatmap_noregion.jpg', map3)

        return map3

    if returnFilteroutLabels:
        F0, region_patch_list, eraseLabels, filterout_labels = gsr.generate_subRegions(img, patchSize, region_patch_ratio, eraseMap, k,
                                                                     minSize, sigma, thresh=thresh,
                                                                     diffResolution=diffResolution, returnFilteroutLabels=returnFilteroutLabels)
    else:
        F0, region_patch_list, eraseLabels = gsr.generate_subRegions(img, patchSize, region_patch_ratio, eraseMap, k, minSize, sigma, thresh=thresh, diffResolution=diffResolution)
    maps2by2 = {}
    region_labels = {}
    for ri in range(len(region_patch_list)):
        r = region_patch_list[ri]
        if sdaePara is not None:
            batchSize = len(r)
            inputShape = (batchSize, 1, sizeRange[0], sizeRange[0])
            sdaePara['inputShape'] = inputShape
        if len(r) != 0:
            # if feaType == 'LBP':
            feaVectors, posVectors = glf.genImgLocalFeas(imgFile, feaType, gridSize, sizeRange,
                                                         gridList=r, sdaePara=sdaePara, withIntensity=withIntensity)
            labels = {}
            if specialType is not None:
                w = wordsFile_s[specialType]
                # print feaVectors.shape
                labelVec = chm.calPatchLabels2(w, feaVectors, k=nk, two_classes=['1', '2'], isH1=True, mk=mk)
                name_s = types[specialType] + '_rest'
                labels[name_s] = labelVec
            else:
                for wi in range(len(wordsFile_s)):
                    w = wordsFile_s[wi]
                    labelVec = chm.calPatchLabels2(w, feaVectors, k=nk, two_classes=['1', '2'], isH1=True, mk=mk)
                    name_s = types[wi] + '_rest'
                    labels[name_s] = labelVec

            for k, v in labels.iteritems():
                v = list(v.flatten())
                if k not in maps2by2:
                    maps2by2[k] = np.zeros((3, F0.shape[0], F0.shape[1]))
                    region_labels[k] = [[], [], []]
                c1 = float(v.count(0)) / float(len(v))
                c2 = float(v.count(1)) / float(len(v))
                cc = float(v.count(2)) / float(len(v))
                cs = np.array([c1, c2, cc])
                cs[np.where(cs < th)] = 0

                maps2by2[k][0][np.where(F0 == ri)] = cs[0]
                maps2by2[k][1][np.where(F0 == ri)] = cs[1]
                maps2by2[k][2][np.where(F0 == ri)] = cs[2]
                if cs[0] > 0:
                    region_labels[k][0].append(ri)
                if cs[1] > 0:
                    region_labels[k][1].append(ri)
                if cs[2] > 0:
                    region_labels[k][2].append(ri)
    if isReturnMaps is not None:
        return maps2by2, region_labels.values()[0][2], F0
    if isSave:
        for c, m in maps2by2.iteritems():
            for ii in range(3):
                imsave('heatmap_' + str(ii) + '.jpg', m[ii, :, :])
            map3 = np.transpose(m, (1, 0, 2)).reshape(440, 440 * 3)
            map3 = np.append(im, map3, axis=1)
            imsave(im_name + '_' + feaType + '_' + c + '_region' + '.jpg', map3)

    if train:
        return F0, region_labels, eraseLabels
    else:
        returnlabels = []
        if is_rotate:
            specialLabels = []
        for i in returnRegionLabels:
            for _, v in region_labels.iteritems():
                returnlabels = returnlabels + v[i]
                if is_rotate:
                    specialLabels = specialLabels + v[0]
        if is_rotate:
            return F0, returnlabels, eraseLabels, specialLabels
        else:
            if returnFilteroutLabels:
                return F0, returnlabels, eraseLabels, filterout_labels
            else:
                return F0, returnlabels, eraseLabels

def showSelectRegions(F, regionLabels, angle=None):
    mm = np.zeros(F.shape)
    for i in regionLabels:
        mm[np.where(F==i)] = 255
    if angle is not None:
        mm = rotate(mm, angle, preserve_range=True)
    plt.figure(1111)
    plt.imshow(mm, cmap='gray')
    # plt.show()

def showMaps3(maps3):
    for c, m in maps3.iteritems():
        map3 = np.transpose(m, (1, 0, 2)).reshape(440, 440 * 3)
        plt.imshow(map3, cmap='gray')
        plt.show()

if __name__ == '__main__':
    paras = {}
    imgFile = '../../Data/labeled2003_38044/N20031221G093121.bmp'
    paras['imgFile'] = imgFile
    # sdae_wordsFile_h1 = '../../Data/Features/type4_SDAEWords_h1.hdf5'
    # sdae_wordsFile_h2 = '../../Data/Features/type4_SDAEWords_h2.hdf5'
    # sdae_wordsFile_h1_diff_mean = '../../Data/Features/type4_SDAEWords_h1_diff_mean.hdf5'
    # sdae_wordsFile_h2_diff_mean = '../../Data/Features/type4_SDAEWords_h2_diff_mean.hdf5'
    # sift_wordsFile_h1 = '../../Data/Features/type4_SIFTWords_h1.hdf5'
    # sift_wordsFile_h2 = '../../Data/Features/type4_SIFTWords_h2.hdf5'
    # lbp_wordsFile_h1 = '../../Data/Features/type4_LBPWords_h1.hdf5'
    # lbp_wordsFile_h2 = '../../Data/Features/type4_LBPWords_h2.hdf5'

    sift_wordsFile_h1 = '../../Data/Features/type4_SIFTWords_h1_s16_600_300_300_300.hdf5'
    sift_wordsFile_h2 = '../../Data/Features/type4_SIFTWords_h2_s16_600_300_300_300.hdf5'
    sdae_wordsFile_h1_d = '../../Data/Features/type4_SDAEWords_h1_diff_mean_s16_600_300_300_300.hdf5'
    sdae_wordsFile_h2_d = '../../Data/Features/type4_SDAEWords_h2_diff_mean_s16_600_300_300_300.hdf5'
    sdae_wordsFile_h1_s = '../../Data/Features/type4_SDAEWords_h1_same_mean_s16_600_300_300_300.hdf5'
    sdae_wordsFile_h2_s = '../../Data/Features/type4_SDAEWords_h2_same_mean_s16_600_300_300_300.hdf5'
    lbp_wordsFile_h1 = '../../Data/Features/type4_LBPWords_h1_s16_600_300_300_300.hdf5'
    lbp_wordsFile_h2 = '../../Data/Features/type4_LBPWords_h2_s16_600_300_300_300.hdf5'
    cascade_wordsFile = '../../Data/Features/type4_cascadeWords_fea4_s16_600_300_300_300_modify.hdf5'
    SIFTFeaFile_reduce = '../../Data/Features/type4_SIFTFeatures_s16_600_300_300_300_reduce.hdf5'
    SDAEFeaFile_reduce_d = '../../Data/Features/type4_SDAEFeas_diff_mean_s16_600_300_300_300_reduce_sameRatio.hdf5'
    LBPFeaFile_reduce = '../../Data/Features/type4_LBPFeatures_s16_600_300_300_300_reduce_sameRatio.hdf5'
    SDAEFeaFile_reduce_s = '../../Data/Features/type4_SDAEFeas_same_mean_s16_600_300_300_300_reduce_sameRatio.hdf5'
    paras['SIFTFeaFile_reduce'] = SIFTFeaFile_reduce
    paras['SDAEFeaFile_reduce_d'] = SDAEFeaFile_reduce_d
    paras['SDAEFeaFile_reduce_s'] = SDAEFeaFile_reduce_s
    paras['LBPFeaFile_reduce'] = LBPFeaFile_reduce
    paras['sift_wordsFile_h1'] = sift_wordsFile_h1
    paras['sift_wordsFile_h2'] = sift_wordsFile_h2
    paras['sdae_wordsFile_h1_d'] = sdae_wordsFile_h1_d
    paras['sdae_wordsFile_h2_d'] = sdae_wordsFile_h2_d
    paras['sdae_wordsFile_h1_s'] = sdae_wordsFile_h1_s
    paras['sdae_wordsFile_h2_s'] = sdae_wordsFile_h2_s
    paras['lbp_wordsFile_h1'] = lbp_wordsFile_h1
    paras['lbp_wordsFile_h2'] = lbp_wordsFile_h2
    paras['cascade_wordsFile'] = cascade_wordsFile
    paras['k'] = 60
    paras['minSize'] = 50
    paras['patchSize'] = np.array([16, 16])
    paras['region_patch_ratio'] = 0.1
    paras['sigma'] = 0.5
    paras['alpha'] = 0.6
    paras['th'] = 0.45
    paras['types'] = ['arc', 'drapery', 'radial', 'hot_spot']
    paras['lbp_wordsFile_s1'] = '../../Data/Features/type4_LBPWords_s1_s16_b300_w500.hdf5'
    paras['lbp_wordsFile_s2'] = '../../Data/Features/type4_LBPWords_s2_s16_b300_w500.hdf5'
    paras['lbp_wordsFile_s3'] = '../../Data/Features/type4_LBPWords_s3_s16_b300_w500.hdf5'
    paras['lbp_wordsFile_s4'] = '../../Data/Features/type4_LBPWords_s4_s16_b300_w500.hdf5'

    paras['sift_wordsFile_s1'] = '../../Data/Features/type4_SIFTWords_s1_s16_b300.hdf5'
    paras['sift_wordsFile_s2'] = '../../Data/Features/type4_SIFTWords_s2_s16_b300.hdf5'
    paras['sift_wordsFile_s3'] = '../../Data/Features/type4_SIFTWords_s3_s16_b300.hdf5'
    paras['sift_wordsFile_s4'] = '../../Data/Features/type4_SIFTWords_s4_s16_b300.hdf5'

    # k = 100
    # minSize = 100
    # patchSize = np.array([16, 16])
    # region_patch_ratio = 0.1
    # sigma = 0.5
    # alpha = 0.6
    # th = 0.4

    eraseMapPath = '../../Data/eraseMap.bmp'
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

    # F0, region_patch_list = gsr.generate_subRegions(imgFile, patchSize, region_patch_ratio, eraseMap, k, minSize, sigma)
    # eraseLabels = set(list(F0[np.where(eraseMap == 1)].flatten()))
    # paras['eraseLabels'] = eraseLabels
    paras['eraseMap'] = eraseMap
    paras['sizeRange'] = (16, 16)
    paras['imResize'] = (256, 256)
    paras['imgSize'] = (440, 440)
    paras['nk'] = 19
    resolution = 1
    gridSize = np.array([resolution, resolution])
    paras['resolution'] = resolution
    paras['gridSize'] = gridSize
    im = np.array(imread(imgFile), dtype='f') / 255
    paras['im'] = im
    im1 = skimage.io.imread(imgFile)
    if len(im1.shape) == 2:
        img = skimage.color.gray2rgb(im1)
    paras['img'] = img

    sdaePara = {}
    # sdaePara['weight'] = '../../Data/autoEncoder/final_0.01.caffemodel'
    # sdaePara['weight'] = '../../Data/autoEncoder/layer_diff_mean_final.caffemodel'
    sdaePara['weight_d'] = '../../Data/autoEncoder/layer_diff_mean_s16_final.caffemodel'
    sdaePara['weight_s'] = '../../Data/autoEncoder/layer_same_mean_s16_final.caffemodel'
    sdaePara['net'] = '../../Data/autoEncoder/test_net.prototxt'
    sdaePara['meanFile'] = '../../Data/patchData_mean_s16.txt'
    sdaePara['patchMean'] = False
    # layerNeuronNum = [28 * 28, 2000, 1000, 500, 128]
    layerNeuronNum = [16 * 16, 1000, 1000, 500, 64]
    sdaePara['layerNeuronNum'] = layerNeuronNum

    paras['sdaePara'] = sdaePara

    paras['feaType'] = 'LBP'
    paras['mk'] = 0
    paras['thresh'] = 0 # calculateThreshold(imgFile)
    paras['withIntensity'] = False
    paras['diffResolution'] = False
    paras['isSave'] = True
    paras['is_rotate'] = False
    paras['specialType'] = 2  # 0: arc, 1: drapery, 2: radial, 3: hot-spot
    paras['returnRegionLabels'] = [0]  # 0: special, 1: rest, 2: common
    paras['train'] = False

    # region_category_map(paras)

    F0, region_labels, eraseLabels = region_special_map(paras)
    print region_labels
    showSelectRegions(F0, region_labels)
    plt.figure(11)
    plt.imshow(im, cmap='gray')
    # plt.show()

    # for showing heat maps without superpixels
    paras['nk'] = 19
    resolution = 5
    gridSize = np.array([resolution, resolution])
    paras['resolution'] = resolution
    paras['gridSize'] = gridSize
    paras['isSave'] = True
    paras['th'] = 0.45
    map3 = region_special_map(paras, no_region=True)
    plt.figure(1)
    plt.imshow(map3, cmap='gray')
    plt.show()