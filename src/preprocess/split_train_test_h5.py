import h5py
import numpy as np
import random
import math

if __name__ == '__main__':
    # patchDataPath = '../../Data/one_in_minute_patch_diff_mean.hdf5'
    # patchTrain = '../../Data/one_in_minute_patch_train_diff_mean.hdf5'
    # patchTest = '../../Data/one_in_minute_patch_test_diff_mean.hdf5'
    # patchTrainList = '../../Data/patchListTrain_diff_mean.txt'
    # patchTestList = '../../Data/patchListTest_diff_mean.txt'
    # patchDataPath = '../../Data/type4_diff_mean_s16.hdf5'
    # patchTrain = '../../Data/type4_train_diff_mean_s16.hdf5'
    # patchTest = '../../Data/type4_test_diff_mean_s16.hdf5'
    # patchTrainList = '../../Data/type4_train_diff_mean_s16.txt'
    # patchTestList = '../../Data/type4_test_diff_mean_s16.txt'
    # patchDataPath = '../../Data/type4_same_mean_s16.hdf5'
    # patchTrain = '../../Data/type4_train_same_mean_s16.hdf5'
    # patchTest = '../../Data/type4_test_same_mean_s16.hdf5'
    # patchTrainList = '../../Data/type4_train_same_mean_s16.txt'
    # patchTestList = '../../Data/type4_test_same_mean_s16.txt'
    patchDataPath = '../../Data/type4_same_mean_s28_special.hdf5'
    patchTrain = '../../Data/type4_train_same_mean_s28_special.hdf5'
    patchTest = '../../Data/type4_test_same_mean_s28_special.hdf5'
    patchTrainList = '../../Data/type4_train_same_mean_s28_special.txt'
    patchTestList = '../../Data/type4_test_same_mean_s28_special.txt'
    train_rate = 0.9
    f = h5py.File(patchDataPath, 'r')

    data = f.get('data')
    label = f.get('label')

    patch_num = data.shape[0]
    train_num = int(patch_num * train_rate)

    idx = range(patch_num)
    random.shuffle(idx)
    train_idx = idx[0:train_num]
    train_idx.sort()
    test_idx = idx[train_num:patch_num]
    test_idx.sort()
    test_num = len(test_idx)

    print 'train num: ' + str(train_num)
    print 'test num: ' + str(test_num)

    f_train = h5py.File(patchTrain, 'w')
    f_test = h5py.File(patchTest, 'w')

    batchSize = 512
    d_tr = f_train.create_dataset('data', (0, data.shape[1], data.shape[2], data.shape[3]), dtype='f',
                           maxshape=(None, data.shape[1], data.shape[2], data.shape[3]))
    l_tr = f_train.create_dataset('label', (0,), dtype='i', maxshape=(None,))

    d_te = f_test.create_dataset('data', (0, data.shape[1], data.shape[2], data.shape[3]), dtype='f',
                           maxshape=(None, data.shape[1], data.shape[2], data.shape[3]))
    l_te = f_test.create_dataset('label', (0,), dtype='i', maxshape=(None,))

    for i in range(int(math.ceil(float(train_num)/float(batchSize)))):
        s = i*batchSize
        e = min((i+1)*batchSize, train_num)
        print 'train: ' + str(e)
        data_train = np.array(data[train_idx[s:e], :, :, :])
        label_train = np.array(label[train_idx[s:e], ]) - 1
        oldNum = d_tr.shape[0]
        newNum = oldNum + data_train.shape[0]

        d_tr.resize(newNum, axis=0)
        d_tr[oldNum:newNum, :, :, :] = data_train
        l_tr.resize(newNum, axis=0)
        l_tr[oldNum:newNum, ] = label_train
    f_train.close()

    for i in range(int(math.ceil(float(test_num)/float(batchSize)))):
        s = i*batchSize
        e = min((i+1)*batchSize, test_num)
        print 'test: ' + str(e)
        data_test = np.array(data[test_idx[s:e], :, :, :])
        label_test = np.array(label[test_idx[s:e], ]) - 1
        oldNum = d_te.shape[0]
        newNum = oldNum + data_test.shape[0]

        d_te.resize(newNum, axis=0)
        d_te[oldNum:newNum, :, :, :] = data_test
        l_te.resize(newNum, axis=0)
        l_te[oldNum:newNum, ] = label_test
    f_test.close()

    f.close()

    with open(patchTrainList, 'w') as ftr:
        ftr.write(patchTrain)
    with open(patchTestList, 'w') as fte:
        fte.write(patchTest)
    ftr.close()
    fte.close()

