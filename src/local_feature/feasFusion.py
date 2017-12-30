from src.local_feature.generateLocalFeatures import genImgLocalFeas
import numpy as np
import h5py
import src.util.paseLabeledFile as plf
import matplotlib.pyplot as plt
import random

def calFusionFea(imgFile, feaTypes, gridSize, sizeRange, gridList=None):
    for feaType in feaTypes:
        feas, positions = genImgLocalFeas(imgFile, feaType, gridSize, sizeRange, HisFeaDim=32, gridList=gridList)
        if gridList is None:
            fusionFeas = np.empty((len(positions), 0))
            gridList = list(positions)
        fusionFeas = np.append(fusionFeas, feas, axis=1)
    return fusionFeas, positions

def calFusionFeaSet(dataFolder, labelFile, classNum, feaTypes, imgType, gridSize, sizeRange, classLabel, saveName):
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
        feaArr = None
        posArr = None
        for name in imgs:
            imgFile = dataFolder+name+imgType
            print imgFile, c
            feaVec, posVec = calFusionFea(imgFile, feaTypes, gridSize, sizeRange)
            if feaArr is None:
                feaArr = np.empty((0, feaVec.shape[1]))
                posArr = np.empty((0, posParaNum))

            feaArr = np.append(feaArr, feaVec, axis=0)
            posArr = np.append(posArr, posVec, axis=0)
        feaSet.create_dataset(c, feaArr.shape, 'f', feaArr)
        posSet.create_dataset(c, posArr.shape, 'i', posArr)
    f.close()
    print saveName+' saved'
    return 0

if __name__ == '__main__':
    confusion_feas = ['LBP', 'His']

    patchSize = 16
    labelFile = '../../Data/type4_b300.txt'
    imagesFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    gridSize = np.array([10, 10])
    sizeRange = (patchSize, patchSize)
    classLabel4 = [['1'], ['2'], ['3'], ['4']]
    saveName = 'type4_LBPHisFeatures_s16_b300.hdf5'
    saveFolder = '../../Data/Features/'
    calFusionFeaSet(imagesFolder, labelFile, 4, confusion_feas, imgType, gridSize, sizeRange, classLabel4,  saveFolder+saveName)

    # imgFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/N20031221G032551.bmp'
    # fusion_feas, positions = calFusionFea(imgFile, confusion_feas, gridSize, sizeRange)
    #
    # _, axes = plt.subplots(20, 1)
    # r = range(fusion_feas.shape[0])
    # print len(r)
    # random.shuffle(r)
    # for i in range(20):
    #     fea = fusion_feas[r[i], :]
    #     axes[i].plot(fea)
    #
    # plt.show()


