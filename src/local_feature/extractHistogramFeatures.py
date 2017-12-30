from skimage import feature
import src.util.paseLabeledFile as plf
import src.util.normalizeVecs as nv
import src.preprocess.esg as esg
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.misc import imread

def histogramOfPatch(patch_arr, num_bins):
    hist, bins = np.histogram(patch_arr.flatten(), bins=range(0, 256 + 1, 256/num_bins))
    return hist, bins


def calImgHisFeatures(imgFile, gridSize, sizeRange, imResize=None, gridList=None, norm=True, HisFeaDim=64):
    # print imgFile
    if imResize:
        gridPatchData, positions, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange, imResize=imResize, gridList=gridList, imNorm=False)
    else:
        gridPatchData, positions, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange, gridList=gridList, imNorm=False)
    feaVecs = np.zeros((len(gridPatchData), HisFeaDim))
    for i in range(len(gridPatchData)):

        feaVec, _ = histogramOfPatch(gridPatchData[i], HisFeaDim)

        feaVecs[i, :] = feaVec
    if norm:
        feaVecs = nv.normalizeVecs(feaVecs)
    return feaVecs, np.array(positions)

def calHisFeaSet(dataFolder, labelFile, classNum, imgType, gridSize, sizeRange, classLabel, saveName, imResize=None):
    HisDim = 64
    posParaNum = 4
    names, labels = plf.parseNL(labelFile)
    if classNum == 4:
        auroraData = plf.arrangeToClasses(names, labels, classNum, classLabel)
    else:
        auroraData, _ = plf.arrangeToClasses(names, labels, classNum, classLabel)

    f = h5py.File(saveName, 'w')
    f.attrs['dataFolder'] = dataFolder
    ad = f.create_group('auroraData')
    for c, imgs in auroraData.iteritems():
        ascii_imgs = [n.encode('ascii', 'ignore') for n in imgs]
        ad.create_dataset(c, (len(ascii_imgs),), 'S10', ascii_imgs)

    feaSet = f.create_group('feaSet')
    posSet = f.create_group('posSet')
    for c, imgs in auroraData.iteritems():
        feaArr = np.empty((0, HisDim))
        posArr = np.empty((0, posParaNum))
        for name in imgs:
            imgFile = dataFolder+name+imgType
            if imResize:
                feaVec, posVec = calImgHisFeatures(imgFile, gridSize, sizeRange, imResize=imResize)
            else:
                feaVec, posVec = calImgHisFeatures(imgFile, gridSize, sizeRange)
            feaArr = np.append(feaArr, feaVec, axis=0)
            posArr = np.append(posArr, posVec, axis=0)
        feaSet.create_dataset(c, feaArr.shape, 'f', feaArr)
        posSet.create_dataset(c, posArr.shape, 'i', posArr)
    f.close()
    print saveName+' saved'
    return 0

if __name__ == '__main__':
    dataFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    gridSize = np.array([10, 10])
    sizeRange = (64, 64)
    # # classLabel3 = [['1'], ['2'], ['3']]
    # # saveName3 = 'type3_LBPFeatures.hdf5'
    # # classNum3 = 3
    # # labelFileType3 = '../../Data/type3_1000_500_500.txt'
    # # saveFolder = '../../Data/Features/'
    # # calLBPFeaSet(dataFolder, labelFileType3, classNum3, imgType, gridSize, sizeRange, classLabel3,
    # #              saveFolder + saveName3)
    #
    classLabel4 = [['1'], ['2'], ['3'], ['4']]
    # # saveName4 = 'type4_LBPFeatures.hdf5'
    # # saveName4 = 'type4_LBPFeatures_s16_600_300_300_300.hdf5'
    saveName4 = 'type4_HisFeatures_s64_b300.hdf5'
    classNum4 = 4
    # # labelFileType4 = '../../Data/type4_1500_500_500_500.txt'
    # # labelFileType4 = '../../Data/type4_600_300_300_300.txt'
    labelFileType4 = '../../Data/type4_300_300_300_300.txt'
    saveFolder = '../../Data/Features/'
    calHisFeaSet(dataFolder, labelFileType4, classNum4, imgType, gridSize, sizeRange, classLabel4,
                 saveFolder + saveName4)


