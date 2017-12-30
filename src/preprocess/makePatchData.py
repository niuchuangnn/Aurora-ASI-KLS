import numpy as np
import sys
sys.path.insert(0, '../../caffe/python')
import h5py
import src.preprocess.esg as esg
import src.util.paseLabeledFile as plf
from src.preprocess.selectPatch import selectSpecialPatch
import os
import random

def makeSpecialPatchData(labelFile, patchSize, wordsFiles_s, gridSize=np.array([10, 10]),
                  filter_radius=3, channels=1, k=1, imgType='.bmp',
                  feaType='LBP', savePath='../../Data/one_in_minute_patch_diff_mean.hdf5',
                  same_mean_file = '../.../Data/patchData_mean_s16.txt',
                  imagesFolder='../../Data/labeled2003_38044/', patchMean=True,
                  saveList='../../Data/patchList_diff_mean.txt', subtract_same_mean=False):
    sizeRange = (patchSize, patchSize)
    [images, labels] = plf.parseNL(labelFile)
    # arragedImages = plf.arrangeToClasses(images, labels, classNum)

    f = h5py.File(savePath, 'w')
    data = f.create_dataset('data', (0, channels, patchSize, patchSize), dtype='f', maxshape=(None, channels, patchSize, patchSize))
    label = f.create_dataset('label', (0, ), dtype='i', maxshape=(None, ))

    if subtract_same_mean:
        if os.path.exists(same_mean_file):
            f_mean = open(same_mean_file, 'r')
            line = f_mean.readline()
            patch_mean = float(line.split()[1])
        else:
            patches_mean = 0
            for i in range(len(images)):
                imf = imagesFolder + images[i] + imgType
                wordsFile = wordsFiles_s[int(labels[i])-1]

                # gridPatchData, gridList, _ = esg.generateGridPatchData(imf, gridSize, sizeRange)
                gridPatchData, gridList = selectSpecialPatch(imf, wordsFile, feaType, gridSize, sizeRange, k, filter_radius=filter_radius)
                patchData = np.array(gridPatchData)
                patches_mean += patchData.mean()
            patch_mean = patches_mean / len(images)
            print 'patch number: ' + str(data.shape[0])
            print 'patch mean: ' + str(patch_mean)
            with open(same_mean_file, 'w') as f2:
                f2.write('patch_mean: ' + str(patch_mean))
            f2.close()
    else:
        patch_mean = 0

    print 'patch_mean: ', patch_mean
    for i in range(len(images)):
        imf = imagesFolder + images[i] + imgType
        print imf
        wordsFile = wordsFiles_s[int(labels[i])-1]
        # gridPatchData, gridList, _ = esg.generateGridPatchData(imf, gridSize, sizeRange)
        gridPatchData, gridList = selectSpecialPatch(imf, wordsFile, feaType, gridSize, sizeRange, k, filter_radius=filter_radius)
        patchData = [p.reshape(channels, patchSize, patchSize) for p in gridPatchData]
        patchData = np.array(patchData) - patch_mean
        if patchMean:
            means = np.mean(np.mean(patchData, axis=-1), axis=-1)
            means = means.reshape(means.shape[0], means.shape[1], 1, 1)
            means = np.tile(means, (1, 1, patchSize, patchSize))
            patchData -= means
        labelData = np.full((len(gridList), ), int(labels[i]), dtype='i')

        oldNum = data.shape[0]
        newNum = oldNum + patchData.shape[0]
        data.resize(newNum, axis=0)
        data[oldNum:newNum, :, :, :] = patchData
        label.resize(newNum, axis=0)
        label[oldNum:newNum, ] = labelData

    f.close()
    print 'make patch data done!'

    with open(saveList, 'w') as f1:
        f1.write(savePath)
    f1.close()
    print saveList + ' saved!'

    return 0

def makePatchData(labelFile, patchSize, gridSize=np.array([10, 10]), imgType='.bmp',
                  channels=1, savePath='../../Data/one_in_minute_patch_diff_mean.hdf5',
                  same_mean_file='../.../Data/patchData_mean_s16.txt',
                  imagesFolder='../../Data/labeled2003_38044/', patchMean=True,
                  saveList='../../Data/patchList_diff_mean.txt', subtract_same_mean=False):
    sizeRange = (patchSize, patchSize)
    [images, labels] = plf.parseNL(labelFile)
    # arragedImages = plf.arrangeToClasses(images, labels, classNum)

    f = h5py.File(savePath, 'w')
    data = f.create_dataset('data', (0, channels, patchSize, patchSize), dtype='f', maxshape=(None, channels, patchSize, patchSize))
    label = f.create_dataset('label', (0, ), dtype='i', maxshape=(None, ))

    if subtract_same_mean:
        patches_mean = 0
        for i in range(len(images)):
            imf = imagesFolder + images[i] + imgType

            gridPatchData, gridList, _ = esg.generateGridPatchData(imf, gridSize, sizeRange)

            patchData = np.array(gridPatchData)
            patches_mean += patchData.mean()
        patch_mean = patches_mean / len(images)
        print 'patch number: ' + str(data.shape[0])
        print 'patch mean: ' + str(patch_mean)
        with open(same_mean_file, 'w') as f2:
            f2.write('patch_mean: ' + str(patch_mean))
        f2.close()
    else:
        patch_mean = 0

    print 'patch_mean: ', patch_mean
    for i in range(len(images)):
        imf = imagesFolder + images[i] + imgType
        print imf

        gridPatchData, gridList, _ = esg.generateGridPatchData(imf, gridSize, sizeRange)

        patchData = [p.reshape(channels, patchSize, patchSize) for p in gridPatchData]
        patchData = np.array(patchData) - patch_mean
        if patchMean:
            means = np.mean(np.mean(patchData, axis=-1), axis=-1)
            means = means.reshape(means.shape[0], means.shape[1], 1, 1)
            means = np.tile(means, (1, 1, patchSize, patchSize))
            patchData -= means
        labelData = np.full((len(gridList), ), int(labels[i]), dtype='i')

        oldNum = data.shape[0]
        newNum = oldNum + patchData.shape[0]
        data.resize(newNum, axis=0)
        data[oldNum:newNum, :, :, :] = patchData
        label.resize(newNum, axis=0)
        label[oldNum:newNum, ] = labelData

    f.close()
    print 'make patch data done!'

    with open(saveList, 'w') as f1:
        f1.write(savePath)
    f1.close()
    print saveList + ' saved!'

    return 0

def balanceLabelData(pataDataFile, trainRatio, savePath_train, savePath_test, classNum=4):
    f = h5py.File(pataDataFile, 'r')
    labels = np.array(f.get('label'),  dtype='f') - 1
    data = np.array(f.get('data'), dtype='f')
    classNums = []
    for i in xrange(classNum):
        classNums.append(len(list(np.where(labels == i)[0])))
    maxBalanceNum = min(classNums)
    # patchNum = labels.shape[0]
    trainNum = int(maxBalanceNum * trainRatio)

    testNum = maxBalanceNum - trainNum
    f_train = h5py.File(savePath_train, 'w')
    f_test = h5py.File(savePath_test, 'w')
    data_train = np.zeros((trainNum*classNum, data.shape[1], data.shape[2], data.shape[3]), dtype='f')
    label_train = np.zeros((trainNum*classNum,), dtype='f')
    data_test = np.zeros((testNum*classNum, data.shape[1], data.shape[2], data.shape[3]), dtype='f')
    label_test = np.zeros((testNum*classNum,), dtype='f')
    ids_save_train = range(trainNum * classNum)
    random.shuffle(ids_save_train)

    for i in xrange(classNum):
        ids_label_i = list(np.where(labels == i)[0])
        random.shuffle(ids_label_i)
        train_ids_label_i = ids_label_i[:trainNum]
        train_ids_label_i.sort()
        test_ids_label_i = ids_label_i[trainNum:(trainNum+testNum)]
        test_ids_label_i.sort()
        train_ids_i = ids_save_train[i*trainNum:(i+1)*trainNum]
        train_ids_i.sort()
        train_data_i = data[train_ids_label_i, :, :, :]
        test_data_i = data[test_ids_label_i, :, :, :]
        data_train[train_ids_i, :, :, :] = train_data_i
        label_train[train_ids_i] = float(i)
        data_test[i*testNum:(i+1)*testNum, :, :, :] = test_data_i
        label_test[i*testNum:(i+1)*testNum] = float(i)

    f_train.create_dataset('data', (trainNum * classNum, data.shape[1], data.shape[2], data.shape[3]), data=data_train, dtype='f')
    f_train.create_dataset('label', (trainNum * classNum,), data=label_train, dtype='f')

    f_test.create_dataset('data', (testNum * classNum, data.shape[1], data.shape[2], data.shape[3]), data=data_test, dtype='f')
    f_test.create_dataset('label', (testNum * classNum,), data=label_test, dtype='f')

    f_train.close()
    f_test.close()

    patchTrainList = savePath_train[:-5] + '.txt'
    patchTestList = savePath_test[:-5] + '.txt'

    with open(patchTrainList, 'w') as ftr:
        ftr.write(savePath_train)
    with open(patchTestList, 'w') as fte:
        fte.write(savePath_test)
    ftr.close()
    fte.close()

    return 0

if __name__ == '__main__':

    labelFile = '../../Data/balanceSampleFrom_one_in_minute.txt'
    labelFile_s = '../../Data/labeled2003_38044_G_selected.txt'

    # labelFile = '../../Data/type4_600_300_300_300.txt'
    imagesFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    gridSize = np.array([10, 10])
    # sizeRange = (30, 30)
    patchSize = 28
    savePatch_same_mean_s16 = '../../Data/type4_same_mean_s16.hdf5'
    saveList_same_mean_s16 = '../../Data/type4_same_mean_s16.txt'
    savePatch_diff_mean_s16 = '../../Data/type4_diff_mean_s16.hdf5'
    saveList_diff_mean_s16 = '../../Data/type4_diff_mean_s16.txt'

    savePatch_same_mean_s28_special = '../../Data/type4_same_mean_s28_special.hdf5'
    saveList_same_mean_s28_special = '../../Data/type4_same_mean_s28_special.txt'
    savePatch_diff_mean_s28_special = '../../Data/type4_diff_mean_s28_special.hdf5'
    saveList_diff_mean_s28_special = '../../Data/type4_diff_mean_s28_special.txt'

    lbp_wordsFile_s1 = '../../Data/Features/type4_LBPWords_s1_s16_300_300_300_300.hdf5'
    lbp_wordsFile_s2 = '../../Data/Features/type4_LBPWords_s2_s16_300_300_300_300.hdf5'
    lbp_wordsFile_s3 = '../../Data/Features/type4_LBPWords_s3_s16_300_300_300_300.hdf5'
    lbp_wordsFile_s4 = '../../Data/Features/type4_LBPWords_s4_s16_300_300_300_300.hdf5'
    wordsFiles_s = [lbp_wordsFile_s1, lbp_wordsFile_s2, lbp_wordsFile_s3, lbp_wordsFile_s4]
    same_mean_file = '../../Data/patchData_mean_s28_special.txt'

    makePatchData(labelFile, patchSize, patchMean=False, subtract_same_mean=True,
                  savePath=savePatch_same_mean_s16, saveList=saveList_same_mean_s16)
    # makePatchData(labelFile, patchSize, patchMean=True, subtract_same_mean=False,
    #               savePath=savePatch_diff_mean_s16, saveList=saveList_diff_mean_s16)

    # makeSpecialPatchData(labelFile_s, patchSize, wordsFiles_s, patchMean=False, subtract_same_mean=True,
    #                      savePath=savePatch_same_mean_s28_special, saveList=saveList_same_mean_s28_special,
    #                      same_mean_file=same_mean_file)

    # savePath_train = '../../Data/type4_same_mean_s28_special_train.hdf5'
    # savePath_test = '../../Data/type4_same_mean_s28_special_test.hdf5'
    # balanceLabelData(savePatch_same_mean_s28_special, 0.9, savePath_train, savePath_test)

    # [images, labels] = plf.parseNL(labelFile)
    #
    # imgFile = imagesFolder + images[0] + imgType
    # gridPatchData, gridList, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange)
    #
    # print gridList[0:5]
    # plf.showGrid(im, gridList[0:5])
    # plt.show()