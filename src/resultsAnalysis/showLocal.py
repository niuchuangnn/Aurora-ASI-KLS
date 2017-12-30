import numpy as np
import src.VisWords.VisWordsAnalysis as vwa
import matplotlib.pyplot as plt
import src.local_feature.extractDSiftFeatures as extSift
import src.util.paseLabeledFile as plf
import h5py
from scipy.misc import imread, imresize
import copy
import sys
sys.path.insert(0, '../../caffe/python')
import caffe
import src.local_feature.autoencoder as AE
import src.preprocess.esg as esg
import src.local_feature.extractSDAEFeatures as extSDAE
import src.local_feature.extractLBPFeatures as extlbp

def calImgPatchLabel(wordsFile, feaVectors):

    fw = h5py.File(wordsFile, 'r')
    w1 = fw.get('/1/words')
    w2 = fw.get('/2/words')
    w1 = np.array(w1)
    w2 = np.array(w2)
    num_words = w1.shape[0]
    patch_num = feaVectors.shape[0]
    dis1 = np.zeros((patch_num, num_words))
    dis2 = np.zeros((patch_num, num_words))

    for v in range(patch_num):
        dis1[v, :] = np.linalg.norm(w1-feaVectors[v], axis=1)
        dis2[v, :] = np.linalg.norm(w2-feaVectors[v], axis=1)

    dis1_min = dis1.min(axis=1)
    dis1_min_idx = dis1.argmin(axis=1)
    dis2_min = dis2.min(axis=1)
    dis2_min_idx = dis2.argmin(axis=1)

    w1_common_idx, w2_common_idx = vwa.calCommonVector(wordsFile)

    labelVoc = np.array(((dis1_min-dis2_min) > 0), dtype='i') # class1: 0, class2: 1, common: 2
    for i in range(patch_num):
        if labelVoc[i] == 0:
            if (w1_common_idx == dis1_min_idx[i]).sum() > 0:
                labelVoc[i] = 2
        if labelVoc[i] == 1:
            if (w2_common_idx == dis2_min_idx[i]).sum() > 0:
                labelVoc[i] = 2

    return labelVoc

def calPatchLabelHierarchy(wordsFile_h1, wordsFile_h2, feaVectors):
    labelVectors_h1 = calImgPatchLabel(wordsFile_h1, feaVectors)
    labelVectors_h2 = calImgPatchLabel(wordsFile_h2, feaVectors)

    fea_c1_idx = np.argwhere(labelVectors_h1 == 0)
    fea_h2_idx = np.argwhere(labelVectors_h1 > 0)
    fea_c1_idx = list(fea_c1_idx.reshape(len(fea_c1_idx)))
    fea_h2_idx = list(fea_h2_idx.reshape(len(fea_h2_idx)))

    labelVectors_h1[fea_h2_idx] = 1
    labelVectors_h2[fea_c1_idx] = 0

    labelVec = labelVectors_h1 + labelVectors_h2
    return labelVec

def showLocalLabel(imgFile, labelVec, posVec, imResize=None, feaType='unknown'):
    im = imread(imgFile)
    if imResize:
        im = imresize(im, imResize)
    # types = [0, 1, 2, 3]
    types = set(labelVec)
    colcors = ['red', 'blue', 'green', 'yellow']
    titles = ['arc', 'drapery', 'radial', 'common']
    for t in types:
        fig, ax = plt.subplots(figsize=(12,12))
        ax.imshow(im, aspect='equal', cmap='gray')
        pos_idx = np.argwhere(labelVec==t)
        for i in range(pos_idx.shape[0]):
            patch = posVec[pos_idx[i, 0], :]
            ax.add_patch(
                plt.Rectangle((patch[1], patch[0]),
                              patch[2], patch[3],
                              fill=True, facecolor=colcors[t],
                              alpha=0.5)
            )
        plt.axis('off')
        plt.title(feaType + titles[t])
        plt.tight_layout()
        plt.draw()

def filterPos(posVec, labelVec, radius, spaceSize):
    pos_num = posVec.shape[0]
    filtered_idx = []
    poses = []
    for i in range(pos_num):
        poses.append([posVec[i, 0], posVec[i, 1]])

    tranSize = range(-radius*spaceSize, (radius+1)*spaceSize, spaceSize)
    # tranSize.remove(0)
    for i in range(pos_num):
        test_pos = poses[i]
        consist_num = 0
        exist_num = 0
        for h in tranSize:
            test_pos_tran_h = copy.deepcopy(test_pos)
            test_pos_tran_h[0] = test_pos[0] + h
            for w in tranSize:
                test_pos_tran_hw = copy.deepcopy(test_pos_tran_h)
                test_pos_tran_hw[1] = test_pos[1] + w
                if test_pos_tran_hw in poses:
                    exist_num += 1
                    neighbor_idx = poses.index(test_pos_tran_hw)
                    if labelVec[i] == labelVec[neighbor_idx]:
                        consist_num += 1
        if (consist_num - 1) < exist_num/2:
            filtered_idx.append(i)
    filtered_pos = np.delete(posVec, filtered_idx, 0)
    filtered_label = np.delete(labelVec, filtered_idx, 0)
    return filtered_pos, filtered_label

if __name__ == '__main__':
    labelFile = '../../Data/balanceSampleFrom_one_in_minute.txt'
    imagesFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    gridSize = np.array([10, 10])
    sizeRange = (28, 28)
    imResize = (256, 256)

    imgName = 'N20031223G125731'
    imgFile = imagesFolder + imgName + imgType

    # ------------show SIFT---------------
    sift_wordsFile_h1 = '../../Data/Features/SIFTWords_h1.hdf5'
    sift_wordsFile_h2 = '../../Data/Features/SIFTWords_h2.hdf5'
    feaVectors, posVectors = extSift.calImgDSift(imgFile, gridSize, sizeRange, imResize=None)

    # calculate single hierarchy
    # labelVectors_h = calImgPatchLabel(wordsFile_h1, feaVectors)

    # show unfiltered
    labelVectors_h = calPatchLabelHierarchy(sift_wordsFile_h1, sift_wordsFile_h2, feaVectors)
    showLocalLabel(imgFile, labelVectors_h, posVectors, imResize=None, feaType='SIFT_')

    # show filtered
    filtered_pos, filtered_label = filterPos(posVectors, labelVectors_h, 1, 10)
    showLocalLabel(imgFile, filtered_label, filtered_pos, imResize=None, feaType='SIFT_filtered_')

    # ---------------show SDEA local results--------------
    sdae_wordsFile_h1 = '../../Data/Features/SDAEWords_h1.hdf5'
    sdae_wordsFile_h2 = '../../Data/Features/SDAEWords_h2.hdf5'

    # define sdae model
    weight = '../../Data/autoEncoder/final_0.01.caffemodel'
    net = '../../Data/autoEncoder/test_net.prototxt'
    meanFile = '../../Data/patchData_mean.txt'
    f_mean = open(meanFile, 'r')
    patch_mean = float(f_mean.readline().split(' ')[1])
    f_mean.close()
    channels = 1
    layerNeuronNum = [28 * 28, 2000, 1000, 500, 128]

    _, gl, _ = esg.generateGridPatchData(imgFile, gridSize, sizeRange)
    batchSize = len(gl)

    inputShape = (batchSize, channels, 28, 28)
    with open(net, 'w') as f1:
        f1.write(str(AE.defineTestNet(inputShape, layerNeuronNum)))

    caffe.set_mode_gpu()
    model = caffe.Net(net, weight, caffe.TEST)

    feaVec, posVec = extSDAE.calImgSDAEFea(imgFile, model, gridSize, sizeRange, channels, patch_mean)
    labelVectors_h = calPatchLabelHierarchy(sdae_wordsFile_h1, sdae_wordsFile_h2, feaVec)
    showLocalLabel(imgFile, labelVectors_h, posVec, imResize=None, feaType='SDAE_')

    filtered_pos, filtered_label = filterPos(posVec, labelVectors_h, 1, 10)
    showLocalLabel(imgFile, filtered_label, filtered_pos, imResize=None, feaType='SDAE_filtered_')

    # ----------------show LBP------------------
    lbp_wordsFile_h1 = '../../Data/Features/LBPWords_h1.hdf5'
    lbp_wordsFile_h2 = '../../Data/Features/LBPWords_h2.hdf5'
    feaVectors, posVectors = extlbp.calImgLBPFeatures(imgFile, gridSize, sizeRange, imResize=None)

    # show unfiltered
    labelVectors_h = calPatchLabelHierarchy(lbp_wordsFile_h1, lbp_wordsFile_h2, feaVectors)
    showLocalLabel(imgFile, labelVectors_h, posVectors, imResize=None, feaType='LBP_')

    # show filtered
    filtered_pos, filtered_label = filterPos(posVectors, labelVectors_h, 1, 10)
    showLocalLabel(imgFile, filtered_label, filtered_pos, imResize=None, feaType='LBP_filtered_')

    plt.show()