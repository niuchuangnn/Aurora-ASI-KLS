import skimage.io
from scipy.misc import imread, imsave
from src.localization.generateSubRegions import generate_subRegions, show_region_patch_grid
import numpy as np
import os
import src.local_feature.generateLocalFeatures as glf
import src.localization.classHeatMap as chm
import src.util.paseLabeledFile as plf
import matplotlib.pyplot as plt
from src.resultsAnalysis.showLocal import filterPos
import src.preprocess.esg as esg

def selectSpecialPatch(imgFile, wordsFile, feaType, gridSize, sizeRange, nk,
                       filter_radius=3, spaceSize=10):
    feaVecs, posVecs = glf.genImgLocalFeas(imgFile, feaType, gridSize, sizeRange)
    # print feaVecs.shape, posVecs.shape
    labelVecs = chm.calPatchLabels2(wordsFile, feaVecs, k=nk, two_classes=['1', '2'], isH1=True)
    posVecs_f, labelVecs_f = filterPos(posVecs, labelVecs, radius=filter_radius, spaceSize=spaceSize)
    specialIDs = list(np.argwhere(labelVecs_f == 0)[:, 0])
    specialPos = list(posVecs_f[specialIDs, :])
    patchData, _, _ = esg.generateGridPatchData(imgFile, gridSize, sizeRange, gridList=specialPos)
    # patchData_arr = np.array(patchData)
    return patchData, specialPos

def threshHoldFilterPatch(im, gridSize, sizeRange):
    patchData, gridList, _ = esg.generateGridPatchData(im, gridSize, sizeRange)
    patchData_arr = np.array(patchData)
    mm = np.mean(np.mean(patchData_arr, axis=2), axis=1)
    ths = mm - 26
    ids = list(np.where(ths > 0))
    print ids
    gridList = [gridList[x] for x in ids[0]]

    return gridList

if __name__ == '__main__':
    imgFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/N20031221G071121.bmp'
    k = 60
    minSize = 100
    patchSize = np.array([28, 28])
    region_patch_ratio = 0.2
    sigma = 0.5
    alpha = 0.6

    imSize = 440
    eraseMap = np.zeros((imSize, imSize))
    radius = imSize / 2
    centers = np.array([219.5, 219.5])
    eraseMapPath = '../../Data/eraseMap.bmp'
    if not os.path.exists(eraseMapPath):
        for i in range(imSize):
            for j in range(imSize):
                if np.linalg.norm(np.array([i, j]) - centers) > radius + 5:
                    eraseMap[i, j] = 1
        imsave(eraseMapPath, eraseMap)
    else:
        eraseMap = imread(eraseMapPath) / 255

    # F0, region_patch_list, _ = generate_subRegions(imgFile, patchSize, region_patch_ratio, eraseMap, k, minSize, sigma)
    # show_region_patch_grid(imgFile, F0, region_patch_list, alpha, eraseMap)

    gridSize = np.array([10, 10])
    sizeRange = (28, 28)
    feaType = 'LBP'
    lbp_wordsFile_s1 = '../../Data/Features/type4_LBPWords_s1_s16_300_300_300_300.hdf5'
    lbp_wordsFile_s2 = '../../Data/Features/type4_LBPWords_s2_s16_300_300_300_300.hdf5'
    lbp_wordsFile_s3 = '../../Data/Features/type4_LBPWords_s3_s16_300_300_300_300.hdf5'
    lbp_wordsFile_s4 = '../../Data/Features/type4_LBPWords_s4_s16_300_300_300_300.hdf5'

    nk = 1
    # feaVecs, posVecs = glf.genImgLocalFeas(imgFile, feaType, gridSize, sizeRange)
    # # print feaVecs.shape, posVecs.shape
    # labelVecs = chm.calPatchLabels2(w, feaVecs, k=nk, two_classes=['1', '2'], isH1=True)
    # posVecs_f, labelVecs_f = filterPos(posVecs, labelVecs, radius=3, spaceSize=10)
    # specialIDs = list(np.argwhere(labelVecs_f == 0)[:, 0])
    # specialPos = list(posVecs_f[specialIDs, :])
    wordsFile_s = [lbp_wordsFile_s1, lbp_wordsFile_s2, lbp_wordsFile_s3, lbp_wordsFile_s4]
    specialType = 2
    w = wordsFile_s[specialType]
    patchData_a, specialPos = selectSpecialPatch(imgFile, w, feaType, gridSize, sizeRange, nk)
    # print labelVecs.shape
    print len(specialPos)
    print len(patchData_a)
    im = imread(imgFile)
    patchData, _, _ = esg.generateGridPatchData(im, gridSize, sizeRange, gridList=specialPos)
    print len(patchData)
    pp = np.array(patchData)
    background_mean = pp.mean()
    print background_mean
    mm = np.mean(np.mean(pp, axis=2), axis=1)
    print mm.shape
    plf.showGrid(im, specialPos)
    plt.show()
    specialList = threshHoldFilterPatch(im, gridSize, sizeRange)
    plf.showGrid(im, specialList)
    plt.show()