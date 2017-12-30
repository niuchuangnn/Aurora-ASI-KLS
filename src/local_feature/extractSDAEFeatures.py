'''
Chuang Niu, niuchuang@stu.xidian.edu.cn
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../fast-rcnn/caffe-fast-rcnn/python')
import caffe
from caffe import layers as L, params as P
import os
import h5py
from caffe.proto import caffe_pb2
import src.preprocess.esg as esg
import src.util.paseLabeledFile as plf
import src.util.normalizeVecs as nv
import src.local_feature.autoencoder as AE

def calImgSDAEFea(imgFile, model, gridSize, sizeRange, channels, patch_mean,
                  gridList=None, imResize=None, patchMean=True, norm=True):
    patchSize = sizeRange[0]
    if imResize:
        gridPatchData, gridList, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange, imResize=imResize,
                                                                gridList=gridList)
    else:
        gridPatchData, gridList, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange, gridList=gridList)
    # gridPatchData, gridList, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange, gridList=gridList)
    patchData = [p.reshape(channels, patchSize, patchSize) for p in gridPatchData]
    patchData = np.array(patchData) - patch_mean
    if patchMean:
        means = np.mean(np.mean(patchData, axis=-1), axis=-1)
        means = means.reshape(means.shape[0], means.shape[1], 1, 1)
        means = np.tile(means, (1, 1, patchSize, patchSize))
        patchData -= means
    labelData = np.full((len(gridList),), int(0), dtype='float32')

    model.set_input_arrays(patchData, labelData)
    out = model.forward()
    out_name = model.blobs.items()[-1][0]
    feaVec = out[out_name]
    posVec = np.array(gridList)
    if norm:
        feaVec = nv.normalizeVecs(feaVec)
    return feaVec, posVec

if __name__ == '__main__':
    dataFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    gridSize = np.array([10, 10])
    sizeRange = (28, 28)
    patchSize = sizeRange[0]
    channels = 1
    posParaNum = 4
    layerNeuronNum = [28 * 28, 1000, 1000, 500, 64]
    SDAEFeaDim = layerNeuronNum[-1]

    # weight = '../../Data/autoEncoder/final_0.01.caffemodel'
    # weight = '../../Data/autoEncoder/layer_diff_mean_final.caffemodel'
    # weight = '../../Data/autoEncoder/layer_same_mean_s16_final.caffemodel'
    # weight = '../../Data/autoEncoder/layer_same_mean_s28_special_final.caffemodel'
    weight = '../../Data/autoEncoder/layer_same_mean_s28_special_final.caffemodel'
    net = '../../Data/autoEncoder/test_net.prototxt'
    img_test = dataFolder + 'N20031221G030001.bmp'
    gridPatchData, gridList, im = esg.generateGridPatchData(img_test, gridSize, sizeRange)
    batchSize = len(gridList)

    inputShape = (batchSize, 1, 28, 28)
    with open(net, 'w') as f1:
        f1.write(str(AE.defineTestNet(inputShape, layerNeuronNum)))

    caffe.set_mode_gpu()
    model = caffe.Net(net, weight, caffe.TEST)

    labelTruth = '../../Data/Alllabel2003_38044.txt'
    # labelFile = '../../Data/type3_1000_500_500.txt'
    # labelFile = '../../Data/type4_1500_500_500_500.txt'
    # labelFile = '../../Data/type4_600_300_300_300.txt'
    labelFile = '../../Data/type4_b500.txt'
    print plf.compareLabeledFile(labelTruth, labelFile)
    # meanFile = '../../Data/patchData_mean.txt'
    # meanFile = '../../Data/patchData_mean_s16.txt'
    meanFile = '../../Data/patchData_mean_s28_special.txt'
    classNum = 4
    # classes = [['1'], ['2'], ['3']]
    classes = [['1'], ['2'], ['3'], ['4']]
    f_mean = open(meanFile, 'r')

    patch_mean = float(f_mean.readline().split(' ')[1])
    f_mean.close()
    # diff mean
    # patch_mean = 0

    print 'patch_mean: ' + str(patch_mean)

    # saveSDAEFeas = '../../Data/Features/type3_SDAEFeas.hdf5'
    # saveSDAEFeas = '../../Data/Features/type4_SDAEFeas.hdf5'
    # saveSDAEFeas = '../../Data/Features/type4_SDAEFeas_diff_mean.hdf5'
    # saveSDAEFeas = '../../Data/Features/type4_SDAEFeas_same_mean_s16_600_300_300_300.hdf5'
    # saveSDAEFeas = '../../Data/Features/type4_SDAEFeas_same_mean_s28_b500_special.hdf5'
    saveSDAEFeas = '../../Data/Features/type4_SDAEFeas_same_mean_s28_b500_special_trained.hdf5'
    # saveSDAEFeas = '../../Data/Features/type4_SDAEFeas_same_mean_s28_b500_special_classification.hdf5'
    [images, labels] = plf.parseNL(labelFile)
    # arrImgs, _ = plf.arrangeToClasses(images, labels, classNum, classes)
    arrImgs = plf.arrangeToClasses(images, labels, classNum, classes)

    for i in arrImgs:
        print i, len(arrImgs[i])

    f = h5py.File(saveSDAEFeas, 'w')
    f.attrs['dataFolder'] = dataFolder
    ad = f.create_group('auroraData')
    for c, imgs in arrImgs.iteritems():
        ascii_imgs = [n.encode('ascii', 'ignore') for n in imgs]
        ad.create_dataset(c, (len(ascii_imgs),), 'S10', ascii_imgs)

    feaSet = f.create_group('feaSet')
    posSet = f.create_group('posSet')
    for c, imgs in arrImgs.iteritems():
        feaArr = np.empty((0, SDAEFeaDim))
        posArr = np.empty((0, posParaNum))
        for name in imgs:
            imgFile = dataFolder + name + imgType
            # if imResize:
            #     feaVec, posVec = calImgDSift(imgFile, gridSize, sizeRange, imResize=imResize)
            # else:
            #     feaVec, posVec = calImgDSift(imgFile, gridSize, sizeRange)
            feaVec, posVec = calImgSDAEFea(imgFile, model, gridSize, sizeRange, channels, patch_mean, patchMean=False)
            # gridPatchData, gridList, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange)
            # patchData = [p.reshape(channels, patchSize, patchSize) for p in gridPatchData]
            # patchData = np.array(patchData) - patch_mean
            # labelData = np.full((len(gridList),), int(0), dtype='float32')
            #
            # model.set_input_arrays(patchData, labelData)
            # out = model.forward()
            # out_name = model.blobs.items()[-1][0]
            # feaVec = out[out_name]
            # posVec = np.array(gridList)

            feaArr = np.append(feaArr, feaVec, axis=0)
            posArr = np.append(posArr, posVec, axis=0)

        feaSet.create_dataset(c, feaArr.shape, 'f', feaArr)
        posSet.create_dataset(c, posArr.shape, 'i', posArr)
    f.close()
    print saveSDAEFeas + ' saved'