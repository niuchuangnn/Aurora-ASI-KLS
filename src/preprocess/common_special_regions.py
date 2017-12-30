import skimage
import numpy as np
import matplotlib.pyplot as plt
import os
from src.localization.region_category_map import region_special_map, showSelectRegions
from src.local_feature.adaptiveThreshold import calculateThreshold
from scipy.misc import imsave, imread
from src.util.paseLabeledFile import parseNL
import h5py

def showRegions(h5File, labelFile, imgFolder='../../Data/all38044JPG/', imgType='.jpg'):
    [imgs, types] = parseNL(labelFile)
    fr = h5py.File(h5File, 'r')
    for n in range(0, len(imgs)):
        name = imgs[n]
        label = types[n]
        print name, label
        im = imread(imgFolder+name+imgType)
        if len(im.shape) == 2:
            img = skimage.color.gray2rgb(im)
        mask = np.array(fr.get(name))

        print set(mask.flatten())

        type_num = 6
        colors = np.random.randint(0, 255, (type_num, 3))

        color_regions = colors[mask]

        alpha = 0.6
        result = (color_regions * alpha + img * (1. - alpha)).astype(np.uint8)
        plt.figure(11)
        plt.imshow(im, cmap='gray')
        plt.figure(111)
        plt.imshow(result)
        plt.show()

def generateRegions():
    paras = {}
    paras['k'] = 60
    paras['minSize'] = 50
    paras['patchSize'] = np.array([16, 16])
    paras['region_patch_ratio'] = 0.1
    paras['sigma'] = 0.5
    paras['alpha'] = 0.6
    paras['types'] = ['arc', 'drapery', 'radial', 'hot_spot']
    paras['lbp_wordsFile_s1'] = '../../Data/Features/type4_LBPWords_s1_s16_b300_w500.hdf5'
    paras['lbp_wordsFile_s2'] = '../../Data/Features/type4_LBPWords_s2_s16_b300_w500.hdf5'
    paras['lbp_wordsFile_s3'] = '../../Data/Features/type4_LBPWords_s3_s16_b300_w500.hdf5'
    paras['lbp_wordsFile_s4'] = '../../Data/Features/type4_LBPWords_s4_s16_b300_w500.hdf5'

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

    paras['eraseMap'] = eraseMap
    paras['sizeRange'] = (16, 16)
    paras['imResize'] = (256, 256)
    paras['imgSize'] = (440, 440)
    paras['nk'] = 1
    resolution = 1
    gridSize = np.array([resolution, resolution])
    paras['resolution'] = resolution
    paras['gridSize'] = gridSize
    paras['sdaePara'] = None
    paras['feaType'] = 'LBP'
    paras['mk'] = 0
    paras['withIntensity'] = False
    paras['diffResolution'] = False
    paras['isSave'] = False
    paras['is_rotate'] = False
    paras['train'] = False

    labelFile = '../../Data/type4_b500.txt'
    imageFolder = '../../Data/all38044JPG/'
    imgType = '.jpg'
    [names, labels] = parseNL(labelFile)
    f = h5py.File('../../Data/regions.hdf5', 'w')

    # imgFile = '../../Data/labeled2003_38044/N20031221G071131.bmp'
    # label = 2

    for i in range(len(names)):
        name = names[i]
        imgFile = imageFolder + name + imgType
        label = labels[i]

        paras['imgFile'] = imgFile
        im = np.array(imread(imgFile), dtype='f') / 255
        paras['im'] = im
        im1 = skimage.io.imread(imgFile)
        if len(im1.shape) == 2:
            img = skimage.color.gray2rgb(im1)
        paras['img'] = img
        paras['thresh'] = calculateThreshold(imgFile)

        paras['specialType'] = int(label) - 1  # 0: arc, 1: drapery, 2: radial, 3: hot-spot
        paras['returnRegionLabels'] = [0]  # 0: special, 1: rest, 2: common
        paras['th'] = 0.45

        F0, region_labels, eraseLabels = region_special_map(paras)

        # save type: 0: common, 1: arc, 2, drapery, 3: radial, 4: hot-spot, 5: rest
        mask = np.zeros(F0.shape, dtype='i')
        mask.fill(5)

        for i in region_labels:
            mask[np.where(F0 == i)] = int(label)

        paras['specialType'] = int(label) - 1  # 0: arc, 1: drapery, 2: radial, 3: hot-spot
        paras['returnRegionLabels'] = [2]  # 0: special, 1: rest, 2: common
        paras['th'] = 0.25
        # paras['thresh'] = 0

        F0, region_labels, eraseLabels, filterout_labels = region_special_map(paras, returnFilteroutLabels=True)
        # print filterout_labels
        region_labels = region_labels + filterout_labels
        for i in region_labels:
            mask[np.where(F0 == i)] = 0

        f.create_dataset(name, mask.shape, dtype='i', data=mask)
        print name + ' saved!'
    f.close()

    return 0

if __name__ == '__main__':

    showRegions('../../Data/regions.hdf5', '../../Data/type4_b500.txt')