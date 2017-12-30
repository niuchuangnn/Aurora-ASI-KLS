import src.preprocess.esg as esg
import src.util.paseLabeledFile as plf
import numpy as np
import dsift
from scipy import misc
import matplotlib.pyplot as plt
import h5py
from src.local_feature.intensityFeature import intensityFeature

posParaNum = 4
# saveName = 'balance500SIFT.hdf5'
saveFolder = '../../Data/Features/'


def calImgDSift(imgFile, gridSize, sizeRange, gridList=None, imResize=None, withIntensity=None, diffResolution=True):
    # print imgFile
    siftFeaDim = 128
    # if withIntensity:
    #     siftFeaDim += 3
    if imResize:
        patches, positions, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange, imResize=imResize, gridList=gridList,
                                                           diffResolution=diffResolution)
    else:
        patches, positions, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange, gridList=gridList,
                                                           diffResolution=diffResolution)
    feaVecs = np.zeros((len(patches), siftFeaDim))
    for i in range(len(patches)):
        patchSize = int(positions[i][-1])
        extractor = dsift.SingleSiftExtractor(patchSize)
        feaVec = extractor.process_image(patches[i])
        feaVecs[i, :] = feaVec
    if withIntensity is True:
        intensityFeas = intensityFeature(gridPatchData=patches, diffResolution=diffResolution)
        feaVecs = np.hstack((feaVecs, intensityFeas))
    return feaVecs, np.array(positions)


def calSIFTFeaSet(dataFolder, labelFile, classNum, imgType, gridSize, sizeRange, classLabel, saveName,
                  imResize=None, diffResolution=True, withIntensity=None):
    names, labels = plf.parseNL(labelFile)
    if classNum == 4:
        auroraData = plf.arrangeToClasses(names, labels, classNum, classLabel)
    else:
        auroraData, _ = plf.arrangeToClasses(names, labels, classNum, classLabel)

    siftFeaDim = 128
    if withIntensity:
        siftFeaDim += 3

    f = h5py.File(saveFolder + saveName, 'w')
    f.attrs['dataFolder'] = dataFolder
    ad = f.create_group('auroraData')
    for c, imgs in auroraData.iteritems():
        ascii_imgs = [n.encode('ascii', 'ignore') for n in imgs]
        ad.create_dataset(c, (len(ascii_imgs),), 'S10', ascii_imgs)

    feaSet = f.create_group('feaSet')
    posSet = f.create_group('posSet')
    for c, imgs in auroraData.iteritems():
        feaArr = np.empty((0, siftFeaDim))
        posArr = np.empty((0, posParaNum))
        for name in imgs:
            imgFile = dataFolder + name + imgType
            if imResize:
                feaVec, posVec = calImgDSift(imgFile, gridSize, sizeRange, imResize=imResize, diffResolution=diffResolution,
                                             withIntensity=withIntensity)
            else:
                feaVec, posVec = calImgDSift(imgFile, gridSize, sizeRange, diffResolution=diffResolution,
                                             withIntensity=withIntensity)
            feaArr = np.append(feaArr, feaVec, axis=0)
            posArr = np.append(posArr, posVec, axis=0)
        feaSet.create_dataset(c, feaArr.shape, 'f', feaArr)
        posSet.create_dataset(c, posArr.shape, 'i', posArr)
    f.close()
    print saveFolder + saveName + ' saved'
    return 0


if __name__ == '__main__':
    dataFolder = '../../Data/labeled2003_38044/'
    # labelFile1 = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/balanceSampleFrom_one_in_minute.txt'
    # classLabel1 = [['1'], ['2'], ['3'], ['4']]
    # saveName1 = 'balance500SIFT.hdf5'
    # names1, labels1 = plf.parseNL(labelFile1)
    # classNum1 = 4
    gridSize = [10, 10]
    # sizeRange = [10, 30]
    sizeRange = [64, 64]
    # imResize = (256, 256)
    imgType = '.bmp'
    # auroraData1 = plf.arrangeToClasses(names1, labels1, classNum1)
    # img = misc.imread(dataFolder + auroraData1['1'][0] + imgType)
    # print img.shape
    #
    # patches1, positions1, im1 = esg.generateGridPatchData(dataFolder + auroraData1['1'][0] + imgType, gridSize,
    #                                                       sizeRange, imResize=imResize)
    #
    # patchSize1 = positions1[110][-1]
    # print patchSize1
    #
    # extractor = dsift.SingleSiftExtractor(patchSize1)
    # feaVec = extractor.process_image(patches1[110])
    #
    # feaVecs1, pos1 = calImgDSift(dataFolder + auroraData1['2'][400] + imgType, gridSize, sizeRange)

    # dataSIFTFeature1 = calSIFTFeaSet(dataFolder, labelFile1, classNum1, imgType, gridSize, sizeRange, classLabel1, saveName1)

    # labelFileType3 = '../../Data/type3_1000_500_500.txt'
    #
    # classLabel3 = [['1'], ['2'], ['3']]
    # saveName3 = 'type3_SIFTFeatures_256.hdf5'
    # classNum3 = 3
    # # names3, lables3 = plf.parseNL(labelFileType3)
    #
    # SIFTFeature3 = calSIFTFeaSet(dataFolder, labelFileType3, classNum3, imgType, gridSize, sizeRange, classLabel3,
    #                              saveName3, imResize=imResize)

    # labelFileType4 = '../../Data/type4_1500_500_500_500.txt'
    # labelFileType4 = '../../Data/type4_600_300_300_300.txt'
    labelFileType4 = '../../Data/type4_300_300_300_300.txt'

    classLabel4 = [['1'], ['2'], ['3'], ['4']]
    # saveName4 = 'type4_SIFTFeatures.hdf5'
    # saveName4 = 'type4_SIFTFeatures_s28.hdf5'
    # saveName4 = 'type4_SIFTFeatures_s16_600_300_300_300.hdf5'
    saveName4 = 'type4_SIFTFeatures_s64_b300.hdf5'
    classNum4 = 4
    # names3, lables3 = plf.parseNL(labelFileType3)

    calSIFTFeaSet(dataFolder, labelFileType4, classNum4, imgType, gridSize, sizeRange, classLabel4,
                                 saveName4, imResize=None, withIntensity=False, diffResolution=False)

    # print pos[110][-1]
    # plt.figure()
    # plt.plot(feaVec[0], 'r')
    # plt.figure()
    # plt.plot(feaVecs[600][0], 'b')
    # plt.show()

    pass
