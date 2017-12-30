import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    lbpFeas = '../../Data/Features/type4_LBPWords_h1.hdf5'
    sdaeFeas = '../../Data/Features/type4_SDAEWords_h1.hdf5'
    siftFeas = '../../Data/Features/type4_SIFTWords_h1.hdf5'

    f = h5py.File(sdaeFeas, 'r')
    for i in f:
        print i
    common1 = np.array(f.get('common_vectors/common_vec_1'))
    common2 = np.array(f.get('common_vectors/common_vec_2'))
    w1 = np.array(f.get('1/words'))
    w2 = np.array(f.get('2/words'))
    y1 = np.zeros((w1.shape[0], ))
    y1[common1] = 2
    y2 = np.ones((w2.shape[0], ))
    y2[common2] = 2
    y = np.append(y1, y2)
    idx1 = np.argwhere(y==0)
    idx2 = np.argwhere(y==1)
    idxc = np.argwhere(y==2)
    # print idxc
    # print w1.max(), w1.min()
    # print w2.shape
    # print w2.max(), w2.min()

    pca1 = PCA()
    pca1.fit(w1)
    pca2 = PCA()
    pca2.fit(w2)

    vars = pca1.explained_variance_
    var_ratios = pca1.explained_variance_ratio_

    # print var_ratios
    print var_ratios[0:10].sum()

    u1 = pca1.components_[0:3, :]
    u2 = pca2.components_[0:3, :]
    # print u1.shape
    w1_pca = w1.dot(u1.T)
    w2_pca = w2.dot(u2.T)
    w_pca = np.append(w1_pca, w2_pca, axis=0)
    # print w_pca.shape
    # print y[:500], y[500:]
    # print w1_pca.min(), w1_pca.max()
    axis_min = int(w1_pca.min() - 2)
    axis_max = int(w1_pca.max() + 2)

    fig = plt.figure(1, figsize=(7, 7))
    plt.clf()
    ax = Axes3D(fig)
    plt.cla()

    # ax.scatter(w_pca[:, 0], w_pca[:, 1], w_pca[:, 2], c=y.astype(np.float))
    ax.scatter(w_pca[idx1, 0], w_pca[idx1, 1], w_pca[idx1, 2], c='blue')
    ax.scatter(w_pca[idx2, 0], w_pca[idx2, 1], w_pca[idx2, 2], c='red')
    ax.scatter(w_pca[idxc, 0], w_pca[idxc, 1], w_pca[idxc, 2], c='yellow')
    ax.legend(labels=['1', '2', 'c'])
    # ax.set_xticks(np.linspace(axis_min, axis_max, 2))
    # ax.set_yticks(np.linspace(axis_min, axis_max, 2))
    # ax.set_zticks(np.linspace(axis_min, axis_max, 2))

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    # ax1.scatter(w_pca[:, 0], w_pca[:, 1], c=y.astype(np.float))
    ax1.scatter(w_pca[idx1, 0], w_pca[idx1, 1], c='blue')
    ax1.scatter(w_pca[idx2, 0], w_pca[idx2, 1], c='red')
    ax1.scatter(w_pca[idxc, 0], w_pca[idxc, 1], c='yellow')
    ax1.legend(labels=['1', '2', 'c'])

    # ax2.scatter(w_pca[:, 0], np.zeros(w_pca[:, 1].shape), c=y.astype(np.float))
    ax2.scatter(w_pca[idx1, 0], np.zeros(w_pca[idx1, 1].shape), c='blue')
    ax2.scatter(w_pca[idx2, 0], np.zeros(w_pca[idx2, 1].shape), c='red')
    ax2.scatter(w_pca[idxc, 0], np.zeros(w_pca[idxc, 1].shape), c='yellow')
    ax2.legend(labels=['1', '2', 'c'])

    plt.show()

