import numpy as np
import src.VisWords.VisWordsAnalysis as vwa
import matplotlib.pyplot as plt
import src.local_feature.extractDSiftFeatures as extSift
import src.util.paseLabeledFile as plf
import h5py
from scipy.misc import imread, imresize
# import copy
# import sys
# sys.path.insert(0, '../../caffe/python')
# import caffe
# import src.local_feature.autoencoder as AE
import src.preprocess.esg as esg
# import src.local_feature.extractSDAEFeatures as extSDAE
import src.local_feature.extractLBPFeatures as extlbp
import src.resultsAnalysis.showLocal as sl

def calLabels(feaVec, fh, cdh, classes, patch_num, num_words, ish1):
    class_num = len(classes)
    dis = np.zeros((class_num, patch_num, num_words))
    dis_min = np.zeros((class_num, patch_num))
    dis_min_idx = np.zeros((class_num, patch_num))

    for i in range(class_num):
        wi = fh.get(classes[i] + '/words')
        wi = np.array(wi)
        for v in range(patch_num):
            dis[i, v, :] = np.linalg.norm(wi - feaVec[v], axis=1)
        dis_min[i, :] = dis[i].min(axis=1)
        dis_min_idx[i, :] = dis[i].argmin(axis=1)

    label_dic = {}
    for i in range(class_num):
        for j in range(i + 1, class_num):
            if ish1:
                common_vec_name_i = 'common_vec_' + str(i+1)
                common_vec_name_j = 'common_vec_' + str(j+1)
                key = str(i) + '_' + str(j) + '_h1'
            else:
                common_vec_name_i = 'common_vec_' + str(i) + str(j) + '_' + str(i)
                common_vec_name_j = 'common_vec_' + str(i) + str(j) + '_' + str(j)
                key = str(i) + '_' + str(j)
            common_idx_i = np.array(cdh.get(common_vec_name_i))
            common_idx_j = np.array(cdh.get(common_vec_name_j))

            labelVec = np.array(((dis_min[i] - dis_min[j]) > 0), dtype='i')  # class_i: 0, class_j: 1, common_ij: 2

            for k in range(patch_num):
                if labelVec[k] == 0:
                    if (common_idx_i == dis_min_idx[i, k]).sum() > 0:
                        labelVec[k] = 2
                if labelVec[k] == 1:
                    if (common_idx_j == dis_min_idx[i, k]).sum() > 0:
                        labelVec[k] = 2
            label_dic[key] = labelVec
    return label_dic

def calPatchLabels(feaVectors, wordsFile_h1, wordsFile_h2):
    fh1 = h5py.File(wordsFile_h1, 'r')
    fh2 = h5py.File(wordsFile_h2, 'r')
    classes_h1 = []
    classes_h2 = []
    for i in fh1:
        classes_h1.append(str(i))
    for i in fh2:
        classes_h2.append(str(i))
    common_h1 = classes_h1.pop(-1)
    common_h2 = classes_h2.pop(-1)
    cd_h1 = fh1.get(common_h1)
    cd_h2 = fh2.get(common_h2)
    patch_num = feaVectors.shape[0]
    num_words = fh1.get(classes_h1[0] + '/words').shape[0]

    label_dic_h1 = calLabels(feaVectors, fh1, cd_h1, classes_h1, patch_num, num_words, ish1=True)
    label_dic_h2 = calLabels(feaVectors, fh2, cd_h2, classes_h2, patch_num, num_words, ish1=False)
    label_dic = dict(label_dic_h1.items() + label_dic_h2.items())

    return label_dic

def show2(imgFile, labelVec, posVec, title, imResize=None, feaType='unknown'):
    im = imread(imgFile)
    if imResize:
        im = imresize(im, imResize)
    # types = [0, 1, 2, 3]
    types = set(labelVec)
    colcors = ['red', 'blue', 'green', 'yellow']
    # titles = ['arc', 'drapery', 'radial', 'common']
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for t in types:
        ax[t].imshow(im, aspect='equal', cmap='gray')
        pos_idx = np.argwhere(labelVec==t)
        for i in range(pos_idx.shape[0]):
            patch = posVec[pos_idx[i, 0], :]
            ax[t].add_patch(
                plt.Rectangle((patch[1], patch[0]),
                              patch[2], patch[3],
                              fill=True, facecolor=colcors[t],
                              alpha=0.5)
            )
        ax[t].axis('off')
    plt.title(feaType + title)
    plt.tight_layout()
    plt.draw()
    return 0

def show2by2(imgFile, pos_vectors, label_vectors):
    for k, v in label_vectors.iteritems():
        show2(imgFile, v, pos_vectors[k], k, feaType='SIFT')
    return 0

if __name__ == '__main__':
    gridSize = np.array([10, 10])
    sizeRange = (28, 28)
    imgType = '.bmp'
    dataFolder = '../../Data/labeled2003_38044/'
    imgName = 'N20031221G052431'
    imgFile = dataFolder + imgName + imgType
    # imgFile = '/home/niuchuang/PycharmProjects/KLSA-auroral-images/Data/N20040101G140132.jpg'
    sift_wordsFile_h1 = '../../Data/Features/type4_SIFTWords_h1.hdf5'
    sift_wordsFile_h2 = '../../Data/Features/type4_SIFTWords_h2.hdf5'
    feaVectors, posVectors = extSift.calImgDSift(imgFile, gridSize, sizeRange, imResize=None)

    ld = calPatchLabels(feaVectors, sift_wordsFile_h1, sift_wordsFile_h2)

    ld_filtered = {}
    pos_filtered = {}
    pos = {}
    for k, v in ld.iteritems():
        pos[k] = posVectors
        kfl = k + '_filtered'
        kfp = k + '_filtered'
        pos_filtered[kfp], ld_filtered[kfl] = sl.filterPos(posVectors, ld[k], 1, 10)

    show2by2(imgFile, pos_filtered, ld_filtered)
    plt.show()