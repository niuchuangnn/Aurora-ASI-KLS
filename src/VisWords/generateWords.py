import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import h5py
import os

def generateWords(featureH5File, groups, saveFile, wordsNum, feaDim=None, miniBatch=None):
    f = h5py.File(featureH5File, 'r')
    f_w = h5py.File(saveFile, 'w')
    feaSet = f.get('feaSet')

    for i in range(len(groups)):
        c = groups[i]
        if isinstance(c, str):
            feas = None
            feas = feaSet.get(c)
            feas = np.array(feas, dtype='float64')
            print 'cluctering class ' + c + ' with shape ' + str(feas.shape)

        if isinstance(c, list):
            feas = np.empty((0, feaDim), dtype='float64')
            for cs in c:
                feat = feaSet.get(cs)
                feat = np.array(feat, dtype='float64')
                feas = np.append(feas, feat, axis=0)
                print 'cluctering class ' + cs + ' with shape ' + str(feas.shape)

        Kmeans = None
        if miniBatch is None:
            Kmeans = KMeans(n_clusters=wordsNum, n_jobs=-1)
        else:
            Kmeans = MiniBatchKMeans(n_clusters=wordsNum, batch_size=5000, max_no_improvement=100)
        Kmeans.fit(feas)
        cluster_centers = Kmeans.cluster_centers_
        inertia = Kmeans.inertia_
        cc = f_w.create_group(str(i+1))
        cc.attrs['inertia'] = inertia
        cc.create_dataset('words', cluster_centers.shape, 'f', cluster_centers)
    print saveFile + ' saved'
    return 0

if __name__ == '__main__':
    saveFolder = '../../Data/Features/'
    groups = {}
    groups[1] = ['1', ['2', '3', '4']]
    groups[2] = ['2', ['1', '3', '4']]
    groups[3] = ['3', ['1', '2', '4']]
    groups[4] = ['4', ['1', '2', '3']]

    # feaTypes = ['SIFT', 'His']
    # wordsNums = [50, 100, 200, 500]#[200, 500, 800]
    # patchSizes = [8, 16, 24, 32, 40, 48, 56, 64]
    # classLabels = [1, 2, 3, 4]
    feaTypes = ['LBPHis']
    wordsNums = [500]  # [200, 500, 800]
    patchSizes = [16]
    classLabels = [1, 2, 3, 4]
    feaDim = {}
    feaDim['LBP'] = 54
    feaDim['SIFT'] = 128
    feaDim['His'] = 64
    feaDim['LBPHis'] = 86

    for feaType in feaTypes:
        for patchSize in patchSizes:
            featureFile = saveFolder + 'type4_' + feaType + 'Features_s' + str(patchSize) + '_b300.hdf5'
            for wordsNum in wordsNums:
                for label in classLabels:

                    # if (feaType == 'LBP') and (wordsNum != 800) and (patchSize == 8):
                    #     print feaType, str(patchSize), str(wordsNum)
                    #     continue
                    # else:
                    saveName = 'type4_' + feaType + 'Words_s' + str(label) + '_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'
                    print saveName

                    if not os.path.exists(saveFolder+saveName):
                        generateWords(featureFile, groups[label], saveFolder+saveName, wordsNum, feaDim=feaDim[feaType])
                    else:
                        print saveFolder + saveName + ' existed'
