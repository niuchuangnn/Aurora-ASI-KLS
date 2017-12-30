import numpy as np
import h5py
import src.local_feature.generateLocalFeatures as glf
import src.preprocess.esg as esg
import src.util.paseLabeledFile as plf

def normalize_feas(feaArr):
    feaslen = np.sqrt(np.sum(feaArr ** 2, axis=1))
    feaArr_n = feaArr / feaslen.reshape((feaslen.size, 1))
    return feaArr_n

def extractCascadeFeatures(img_file, sift_u, lbp_u, sdae_u_d, sdae_u_s, gridList, gridSize, sizeRange, sdaePara):

    feas_sift, pos = glf.genImgLocalFeas(img_file, 'SIFT', gridSize, sizeRange, u_reduce=sift_u, gridList=gridList)
    feas_lbp, pos = glf.genImgLocalFeas(img_file, 'LBP', gridSize, sizeRange, u_reduce=lbp_u, gridList=gridList)
    sdaePara['weight'] = sdaePara['weight_s']
    sdaePara['patchMean'] = False
    feas_sdae_s, pos = glf.genImgLocalFeas(img_file, 'SDAE', gridSize, sizeRange, sdaePara=sdaePara,
                                                  u_reduce=sdae_u_s, gridList=gridList)

    sdaePara['weight'] = sdaePara['weight_d']
    sdaePara['patchMean'] = True
    feas_sdae_d, pos = glf.genImgLocalFeas(img_file, 'SDAE', gridSize, sizeRange, sdaePara=sdaePara,
                                                  u_reduce=sdae_u_d, gridList=gridList)

    feas_cascade = np.column_stack((feas_sift, feas_lbp, feas_sdae_d, feas_sdae_s))

    return feas_cascade, pos

def calCascadeFeaSet(dataFolder, labelFile, siftFeaFile_reduce, lbpFeaFile_reduce, sdaeFeaFile_reduce_d,
                  sdaeFeaFile_reduce_s,classNum, imgType, gridSize, sizeRange, classLabel, sdaePara,
                  saveName, saveFolder='../../Data/Features/'):
    sift_f = h5py.File(siftFeaFile_reduce, 'r')
    sdae_f_d = h5py.File(sdaeFeaFile_reduce_d, 'r')
    sdae_f_s = h5py.File(sdaeFeaFile_reduce_s, 'r')
    lbp_f = h5py.File(lbpFeaFile_reduce, 'r')

    names, labels = plf.parseNL(labelFile)
    if classNum == 4:
        auroraData = plf.arrangeToClasses(names, labels, classNum, classLabel)
    else:
        auroraData, _ = plf.arrangeToClasses(names, labels, classNum, classLabel)

    f = h5py.File(saveFolder + saveName, 'w')
    f.attrs['dataFolder'] = dataFolder
    ad = f.create_group('auroraData')
    for c, imgs in auroraData.iteritems():
        ascii_imgs = [n.encode('ascii', 'ignore') for n in imgs]
        ad.create_dataset(c, (len(ascii_imgs),), 'S10', ascii_imgs)

    feaSet = f.create_group('feaSet')
    posSet = f.create_group('posSet')
    for c, imgs in auroraData.iteritems():
        # sift_u = np.array(sift_f.get('uSet/'+c))
        # lbp_u = np.array(lbp_f.get('uSet/'+c))
        # sdae_u_d = np.array(sdae_f_d.get('uSet/'+c))
        # sdae_u_s = np.array(sdae_f_s.get('uSet/'+c))
        sift_u = np.array(sift_f.get('uSet/u'))
        lbp_u = np.array(lbp_f.get('uSet/u'))
        sdae_u_d = np.array(sdae_f_d.get('uSet/u'))
        sdae_u_s = np.array(sdae_f_s.get('uSet/u'))
        imgFile = dataFolder + imgs[0] + imgType
        _, gl, _ = esg.generateGridPatchData(imgFile, gridSize, sizeRange)
        feaVec, posVec = extractCascadeFeatures(imgFile, sift_u, lbp_u, sdae_u_d, sdae_u_s, gl, gridSize, sizeRange, sdaePara)
        feaArr = np.empty((0, feaVec.shape[1]))
        posArr = np.empty((0, posVec.shape[1]))
        for name in imgs:
            imgFile = dataFolder + name + imgType
            batchSize = len(gl)
            inputShape = (batchSize, 1, sizeRange[0], sizeRange[0])
            sdaePara['inputShape'] = inputShape
            feaVec, posVec = extractCascadeFeatures(imgFile, sift_u, lbp_u, sdae_u_d, sdae_u_s, gl, gridSize, sizeRange, sdaePara)
            feaArr = np.append(feaArr, feaVec, axis=0)
            posArr = np.append(posArr, posVec, axis=0)
        feaSet.create_dataset(c, feaArr.shape, 'f', feaArr)
        posSet.create_dataset(c, posArr.shape, 'i', posArr)
    f.close()
    print saveFolder + saveName + ' saved'
    return 0

if __name__ == '__main__':
    # siftFeaFile_reduce = '../../Data/Features/type4_SIFTFeatures_reduce.hdf5'
    # SDAEFeaFile_reduce = '../../Data/Features/type4_SDAEFeas_reduce_sameRatio.hdf5'
    # LBPFeaFile_reduce = '../../Data/Features/type4_LBPFeatures_reduce_sameRatio.hdf5'
    # cascade_feas_save_file = '../../Data/Features/type4_cascadeFeatures.hdf5'

    SIFTFeaFile_reduce = '../../Data/Features/type4_SIFTFeatures_s16_600_300_300_300_reduce.hdf5'
    SDAEFeaFile_reduce_d = '../../Data/Features/type4_SDAEFeas_diff_mean_s16_600_300_300_300_reduce_sameRatio.hdf5'
    LBPFeaFile_reduce = '../../Data/Features/type4_LBPFeatures_s16_600_300_300_300_reduce_sameRatio.hdf5'
    SDAEFeaFile_reduce_s = '../../Data/Features/type4_SDAEFeas_same_mean_s16_600_300_300_300_reduce_sameRatio.hdf5'

    # labelFile = '../../Data/balanceSampleFrom_one_in_minute.txt'
    imagesFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    gridSize = np.array([10, 10])
    sizeRange = (16, 16)
    labelFileType4 = '../../Data/type4_600_300_300_300.txt'

    classLabel4 = [['1'], ['2'], ['3'], ['4']]
    saveName4 = 'type4_cascadeFeatures4_s16_600_300_300_300.hdf5'
    classNum4 = 4
    # imgName = 'N20031223G125731'
    # imgFile = imagesFolder + imgName + imgType
    # _, gl, _ = esg.generateGridPatchData(imgFile, gridSize, sizeRange)
    #
    # feas_cas = extractCascadeFeatures(imgFile, SIFTFeaFile_reduce, LBPFeaFile_reduce, SDAEFeaFile_reduce_d,
    #                                   SDAEFeaFile_reduce_s, gl)

    # print feas_cas.max(), feas_cas.min()
    # define SDAE parameters
    sdaePara = {}
    sdaePara['weight_s'] = '../../Data/autoEncoder/layer_same_mean_s16_final.caffemodel'
    sdaePara['weight_d'] = '../../Data/autoEncoder/layer_diff_mean_s16_final.caffemodel'
    sdaePara['net'] = '../../Data/autoEncoder/test_net.prototxt'
    sdaePara['meanFile'] = '../../Data/patchData_mean_s16.txt'

    # channels = 1
    layerNeuronNum = [16 * 16, 1000, 1000, 500, 64]
    sdaePara['layerNeuronNum'] = layerNeuronNum

    calCascadeFeaSet(imagesFolder, labelFileType4, SIFTFeaFile_reduce, LBPFeaFile_reduce, SDAEFeaFile_reduce_d,
                     SDAEFeaFile_reduce_s, classNum4, imgType, gridSize, sizeRange, classLabel4, sdaePara, saveName4)