from skimage import feature
import src.util.paseLabeledFile as plf
import src.util.normalizeVecs as nv
import src.preprocess.esg as esg
import numpy as np
import h5py
import matplotlib.pyplot as plt
from src.local_feature.intensityFeature import intensityFeature

def calImgLBPFeatures(imgFile, gridSize, sizeRange, imResize=None, gridList=None, norm=True, withIntensity=None,
                      diffResolution=False):
    # print imgFile
    P1 = 8
    P2 = 16
    P3 = 24
    R1 = 1
    R2 = 2
    R3 = 3
    if imResize:
        gridPatchData, positions, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange, imResize=imResize, gridList=gridList)
    else:
        gridPatchData, positions, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange, gridList=gridList)
    LBPFeaDim = 10+18+26
    feaVecs = np.zeros((len(gridPatchData), LBPFeaDim))
    for i in range(len(gridPatchData)):
        LBP_img_R1P8 = feature.local_binary_pattern(gridPatchData[i], P1, R1, method='uniform')
        LBP_img_R2P16 = feature.local_binary_pattern(gridPatchData[i], P2, R2, method='uniform')
        LBP_img_R3P24 = feature.local_binary_pattern(gridPatchData[i], P3, R3, method='uniform')

        lbp_bin_num_R1P8 = P1 + 2
        lbp_hist_R1P8, lbp_bins_R1P8 = np.histogram(LBP_img_R1P8.flatten(), bins=range(lbp_bin_num_R1P8 + 1))

        lbp_bin_num_R2P16 = P2 + 2
        lbp_hist_R2P16, lbp_bins_R2P16 = np.histogram(LBP_img_R2P16.flatten(), bins=range(lbp_bin_num_R2P16 + 1))

        lbp_bin_num_R3P24 = P3 + 2
        lbp_hist_R3P24, lbp_bins_R3P24 = np.histogram(LBP_img_R3P24.flatten(), bins=range(lbp_bin_num_R3P24 + 1))

        feaVec = np.array(list(lbp_hist_R1P8) + list(lbp_hist_R2P16) + list(lbp_hist_R3P24))

        feaVecs[i, :] = feaVec
    if norm:
        feaVecs = nv.normalizeVecs(feaVecs)
    # print withIntensity
    # print feaVecs.shape
    if withIntensity is True:
        intensityFeas = intensityFeature(gridPatchData=gridPatchData, diffResolution=diffResolution)
        feaVecs = np.hstack((feaVecs, intensityFeas))
    return feaVecs, np.array(positions)

def calLBPFeaSet(dataFolder, labelFile, classNum, imgType, gridSize, sizeRange, classLabel, saveName,
                 imResize=None, withIntensity=None, diffResolution=True):
    LBPFeaDim = 10 + 18 + 26
    if withIntensity:
        LBPFeaDim += 3
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
        feaArr = np.empty((0, LBPFeaDim))
        posArr = np.empty((0, posParaNum))
        for name in imgs:
            imgFile = dataFolder+name+imgType
            if imResize:
                feaVec, posVec = calImgLBPFeatures(imgFile, gridSize, sizeRange, imResize=imResize,
                                                   withIntensity=withIntensity, diffResolution=diffResolution)
            else:
                feaVec, posVec = calImgLBPFeatures(imgFile, gridSize, sizeRange, withIntensity=withIntensity,
                                                   diffResolution=diffResolution)
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
    # classLabel3 = [['1'], ['2'], ['3']]
    # saveName3 = 'type3_LBPFeatures.hdf5'
    # classNum3 = 3
    # labelFileType3 = '../../Data/type3_1000_500_500.txt'
    # saveFolder = '../../Data/Features/'
    # calLBPFeaSet(dataFolder, labelFileType3, classNum3, imgType, gridSize, sizeRange, classLabel3,
    #              saveFolder + saveName3)

    classLabel4 = [['1'], ['2'], ['3'], ['4']]
    # saveName4 = 'type4_LBPFeatures.hdf5'
    # saveName4 = 'type4_LBPFeatures_s16_600_300_300_300.hdf5'
    saveName4 = 'type4_LBPFeatures_s64_b300.hdf5'
    classNum4 = 4
    # labelFileType4 = '../../Data/type4_1500_500_500_500.txt'
    # labelFileType4 = '../../Data/type4_600_300_300_300.txt'
    labelFileType4 = '../../Data/type4_300_300_300_300.txt'
    saveFolder = '../../Data/Features/'
    calLBPFeaSet(dataFolder, labelFileType4, classNum4, imgType, gridSize, sizeRange, classLabel4,
                 saveFolder + saveName4, withIntensity=True, diffResolution=True)

    # posParaNum = 4
    # P1 = 8
    # P2 = 16
    # P3 = 24
    # R1 = 1
    # R2 = 2
    # R3 = 3
    # B = 10
    #
    # img_test = dataFolder + 'N20031221G030001.bmp'
    # gridPatchData, gridList, im = esg.generateGridPatchData(img_test, gridSize, sizeRange, imNorm=False)
    #
    # # LBP_img = feature.local_binary_pattern(im, P, R, method='uniform')
    # # LBP_var = feature.local_binary_pattern(im, P, R, method='var')
    #
    # LBP_img_R1P8 = feature.local_binary_pattern(gridPatchData[0], P1, R1, method='uniform')
    # LBP_var_R1P8 = feature.local_binary_pattern(gridPatchData[0], P1, R1, method='var')
    #
    # LBP_img_R2P16 = feature.local_binary_pattern(gridPatchData[0], P2, R2, method='uniform')
    # LBP_var_R2P16 = feature.local_binary_pattern(gridPatchData[0], P2, R2, method='var')
    #
    # LBP_img_R3P24 = feature.local_binary_pattern(gridPatchData[0], P3, R3, method='uniform')
    # LBP_var_R3P24 = feature.local_binary_pattern(gridPatchData[0], P3, R3, method='var')

    # gridPatchData_lbp, gridList_lbp, im_lbp = esg.generateGridPatchData(LBP_img, gridSize, sizeRange, imNorm=False)

    # print np.sum(LBP_img_p==gridPatchData_lbp[150])

    # plt.figure(1)
    # plt.plot(LBP_img_p.flatten(), 'r')
    # plt.plot(gridPatchData_lbp[100].flatten(), 'b')

    # plt.figure(1)
    # gif, (ax1,ax2) = plt.subplots(1, 2)
    # ax1.imshow(LBP_img, cmap='gray')
    # ax2.imshow(LBP_var, cmap='gray')
    # ax1.axis('off')
    # ax2.axis('off')
    # plt.show()

    # print LBP_img.shape
    # print LBP_img
    # print LBP_var.shape
    # print LBP_var

    # nan_map = np.isnan(LBP_var_R2P16)
    # nan_pos = np.argwhere(nan_map == True)
    # LBP_var_R2P16[list(nan_pos[:, 0]), list(nan_pos[:, 1])] = 0
    # print LBP_var

    # var_sum = np.sum(LBP_var_R2P16)
    # bin_step = var_sum / B
    # print var_sum, bin_step
    # var_bins = np.linspace(0, var_sum, num=B+1)

    # var_hist, var_bins = np.histogram(LBP_var_R2P16.flatten(), bins=var_bins)
    # print var_hist
    # print var_bins

    # lbp_bin_num_R1P8 = P1 + 2
    # lbp_hist_R1P8, lbp_bins_R1P8 = np.histogram(LBP_img_R1P8.flatten(), bins=range(lbp_bin_num_R1P8 + 1))
    # print lbp_hist_R1P8, len(lbp_hist_R1P8), sum(lbp_hist_R1P8)
    # print lbp_bins_R1P8
    #
    # lbp_bin_num_R2P16 = P2 + 2
    # lbp_hist_R2P16, lbp_bins_R2P16 = np.histogram(LBP_img_R2P16.flatten(), bins=range(lbp_bin_num_R2P16+1))
    # print lbp_hist_R2P16, len(lbp_hist_R2P16), sum(lbp_hist_R2P16)
    # print lbp_bins_R2P16
    #
    # lbp_bin_num_R3P24 = P3 + 2
    # lbp_hist_R3P24, lbp_bins_R3P24 = np.histogram(LBP_img_R3P24.flatten(), bins=range(lbp_bin_num_R3P24 + 1))
    # print lbp_hist_R3P24, len(lbp_hist_R3P24), sum(lbp_hist_R3P24)
    # print lbp_bins_R3P24
    #
    # lbp_fea = list(lbp_hist_R1P8) + list(lbp_hist_R2P16) + list(lbp_hist_R3P24)
    # print lbp_fea, len(lbp_fea)
    #
    # feaVec = np.array(lbp_fea)
    # print feaVec.shape
    #
    # feaVecs, poses = calImgLBPFeatures(dataFolder+img_test, gridSize, sizeRange)
    # print feaVecs.shape

