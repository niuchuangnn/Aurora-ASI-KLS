import h5py
import numpy as np
import matplotlib.pyplot as plt
import random


def normalize_feas(feaArr):
    feaslen = np.sqrt(np.sum(feaArr ** 2, axis=1))
    feaArr_n = feaArr / feaslen.reshape((feaslen.size, 1))
    return feaArr_n

if __name__ == '__main__':
    # lbpFeas = '../../Data/Features/type4_LBPWords_h1_reduce_sameRatio.hdf5'
    # sdaeFeas = '../../Data/Features/type4_SDAEWords_h1_reduce_sameRatio.hdf5'
    # siftFeas = '../../Data/Features/type4_SIFTWords_h1_reduce.hdf5'
    #
    # f = h5py.File(sdaeFeas, 'r')
    # w1 = np.array(f.get('1/words'))
    # w1_n = normalize_feas(w1)
    # print w1.shape
    # print w1.max(), w1.min()
    # print np.argwhere(w1 < 1)
    # print w1_n.shape
    # print w1_n.max(), w1_n.min()
    # print np.argwhere(w1_n < 1)

    lbpFeas = '../../Data/Features/type4_LBPFeatures_s16_b300.hdf5'
    hisFeas = '../../Data/Features/type4_HisFeatures_s16_b300.hdf5'
    sdaeFeas = '../../Data/Features/type4_SDAEFeas_diff_mean_s16_600_300_300_300.hdf5'
    siftFeas = '../../Data/Features/type4_SIFTFeatures_s16_600_300_300_300.hdf5'

    f = h5py.File(lbpFeas, 'r')
    w1 = np.array(f.get('feaSet/1'))
    print w1[0, :]
    print w1.shape
    print w1.max(), w1.min()
    _, axes = plt.subplots(20, 2)
    r = range(300000)
    random.shuffle(r)
    for i in range(20):
        axes[i, 0].plot(w1[r[i], :])
        # axes[i, 1].plot(w1_n[r[i], :])

    plt.show()