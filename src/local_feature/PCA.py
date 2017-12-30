import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D

def calPcaDimReduceU(featureH5File, groups, feaDim=128):
    f = h5py.File(featureH5File, 'a')
    feaSet = f.get('feaSet')
    for i in range(len(groups)):
        c = groups[i]

        if isinstance(c, list):
            ci = 'g_'
            for cii in c:
                ci = ci + cii

            if 'pca/' + ci + '/components' in f:
                print 'pca/' + ci + '/components' + ' exist!'
                continue
            else:
                if 'pca/' + ci in f:
                    pca_group = f.get('pca/' + ci)
                else:
                    pca_group = f.create_group('pca/' + ci)

            feas = np.empty((0, feaDim), dtype='float64')
            for cs in c:
                feat = feaSet.get(cs)
                feat = np.array(feat, dtype='float64')
                feas = np.append(feas, feat, axis=0)
                print 'calculating of class ' + cs + ' with shape ' + str(feas.shape)

        if isinstance(c, str):
            if 'pca/' + c + '/components' in f:
                print 'pca/' + c + '/components' + ' exist!'
                continue
            else:
                if 'pca/' + c in f:
                    pca_group = f.get('pca/' + c)
                else:
                    pca_group = f.create_group('pca/' + c)

            feas = None
            feas = feaSet.get(c)
            feas = np.array(feas, dtype='float64')
            print 'calculating pca of class ' + c + ' with shape ' + str(feas.shape)


        pca = None
        pca = PCA()
        pca.fit(feas)
        # if ratio_num > 1:
        #     keep_components = ratio_num
        # else:
        #     ratios_acc = np.array([var_ratios[:x] for x in range(1, len(var_ratios)+1)])
        #     keep_components = np.argwhere(ratios_acc >= ratio_num) + 1

        # u = pca.components_[:keep_components, :]
        ratios = pca.explained_variance_ratio_
        u = pca.components_
        pca_group.create_dataset('ratios', ratios.shape, 'f', ratios)
        pca_group.create_dataset('components', u.shape, 'f', u)
    f.close()
    print 'pca U saved'
    return 0

def reduceDimension(feaFile, dim_ratio, saveFile, isH1=False):
    f = h5py.File(feaFile, 'r')
    if 'pca' not in f:
        print 'pca not calculated, please calculate pac first!'
    else:
        f_r = h5py.File(saveFile, 'w')
        feaSet = f.get('feaSet')
        feaSet_reduce = f_r.create_group('feaSet')
        uSet = f_r.create_group('uSet')
        if dim_ratio > 1:
            keep_components = dim_ratio
        else:
            ratios = f.get('pca/g_1234/ratios')
            ratios_acc = np.array([ratios[:x].sum() for x in range(1, len(ratios) + 1)])
            keep_components = np.argwhere(ratios_acc >= dim_ratio).min() + 1
        u = f.get('pca/g_1234/components')
        u_reduce = u[:keep_components, :]
        uSet.create_dataset('u', u_reduce.shape, 'f', u_reduce)
        for c in feaSet:
            # u = f.get('pca/'+c+'/components')

            # if dim_ratio > 1:
            #     keep_components = dim_ratio
            # else:
            #     ratios = f.get('pca/'+c+'/ratios')
            #     ratios_acc = np.array([ratios[:x].sum() for x in range(1, len(ratios) + 1)])
            #     keep_components = np.argwhere(ratios_acc >= dim_ratio).min() + 1

            feas = feaSet.get(c)
            feas = np.array(feas, dtype='float64')
            # u_reduce = u[:keep_components, :]
            feas_reduce = feas.dot(u_reduce.T)

            feaSet_reduce.create_dataset(c, feas_reduce.shape, 'f', feas_reduce)
            # uSet.create_dataset(c, u_reduce.shape, 'f', u_reduce)

            if isH1:
                if int(c) > 1:
                    u_h1 = f.get('pca/g_234/components')
                    if dim_ratio > 1:
                        keep_components = dim_ratio
                    else:
                        ratios = f.get('pca/' + c + '/ratios')
                        ratios_acc = np.array([ratios[:x].sum() for x in range(1, len(ratios) + 1)])
                        keep_components = np.argwhere(ratios_acc >= dim_ratio).min() + 1
                    u_reduce_h1 = u_h1[:keep_components, :]
                    feas_reduce_h1 = feas.dot(u_reduce_h1.T)
                    feaSet_reduce.create_dataset('g_'+c, feas_reduce_h1.shape, 'f', feas_reduce_h1)
                    uSet.create_dataset('g_'+c, u_reduce_h1.shape, 'f', u_reduce_h1)
    return 0

def reduceVecFeasDim(feaFile, feaVec, dim_ratio):
    f = h5py.File(feaFile, 'r')
    if 'pca/components' not in f:
        print 'pca not calculated, please calculate pac first!'
    else:
        u = f.get('pca/components')
        if dim_ratio > 1:
            keep_components = dim_ratio
        else:
            ratios = f.get('pca/ratios')
            ratios_acc = np.array([ratios[:x].sum() for x in range(1, len(ratios) + 1)])
            keep_components = np.argwhere(ratios_acc >= dim_ratio).min() + 1

        u_reduce = u[:keep_components, :]
        feas_reduce = feaVec.dot(u_reduce.T)
    return feas_reduce

if __name__ == '__main__':
    # SIFTFeaFile = '../../Data/Features/type4_SIFTFeatures.hdf5'
    # SDAEFeaFile = '../../Data/Features/type4_SDAEFeas.hdf5'
    # LBPFeaFile = '../../Data/Features/type4_LBPFeatures.hdf5'
    SIFTFeaFile = '../../Data/Features/type4_SIFTFeatures_s16_600_300_300_300.hdf5'
    SDAEFeaFile = '../../Data/Features/type4_SDAEFeas_diff_mean_s16_600_300_300_300.hdf5'
    LBPFeaFile = '../../Data/Features/type4_LBPFeatures_s16_600_300_300_300.hdf5'
    SDAEFeaFile_s = '../../Data/Features/type4_SDAEFeas_same_mean_s16_600_300_300_300.hdf5'

    groups = [['1', '2', '3', '4']]

    calPcaDimReduceU(LBPFeaFile, groups, feaDim=54)
    calPcaDimReduceU(SIFTFeaFile, groups, feaDim=128)
    calPcaDimReduceU(SDAEFeaFile, groups, feaDim=64)
    calPcaDimReduceU(SDAEFeaFile_s, groups, feaDim=64)

    # groups_h1 = [['2', '3', '4']]
    # calPcaDimReduceU(LBPFeaFile, groups_h1, feaDim=54)
    # calPcaDimReduceU(SIFTFeaFile, groups_h1, feaDim=128)
    # calPcaDimReduceU(SDAEFeaFile, groups_h1, feaDim=64)
    # calPcaDimReduceU(SDAEFeaFile_s, groups_h1, feaDim=64)

    # SIFTFeaFile_reduce = '../../Data/Features/type4_SIFTFeatures_reduce.hdf5'
    # SDAEFeaFile_reduce = '../../Data/Features/type4_SDAEFeas_reduce_sameRatio.hdf5'
    # LBPFeaFile_reduce = '../../Data/Features/type4_LBPFeatures_reduce_sameRatio.hdf5'

    SIFTFeaFile_reduce = '../../Data/Features/type4_SIFTFeatures_s16_600_300_300_300_reduce.hdf5'
    SDAEFeaFile_reduce = '../../Data/Features/type4_SDAEFeas_diff_mean_s16_600_300_300_300_reduce_sameRatio.hdf5'
    LBPFeaFile_reduce = '../../Data/Features/type4_LBPFeatures_s16_600_300_300_300_reduce_sameRatio.hdf5'
    SDAEFeaFile_reduce_s = '../../Data/Features/type4_SDAEFeas_same_mean_s16_600_300_300_300_reduce_sameRatio.hdf5'

    SIFTReduceDim = 64
    f_sift = h5py.File(SIFTFeaFile, 'r')
    same_ratio = 0
    for i in f_sift.get('pca'):
        ratios_sift = np.array(f_sift.get('pca/'+str(i) + '/ratios'))
        ratios_sift_acc = np.array([ratios_sift[:x].sum() for x in range(1, len(ratios_sift)+1)])
        same_ratio += ratios_sift_acc[SIFTReduceDim-1]
    same_ratio /= len(f_sift.get('pca'))
    reduceDimension(SIFTFeaFile, SIFTReduceDim, SIFTFeaFile_reduce)
    reduceDimension(LBPFeaFile, same_ratio, LBPFeaFile_reduce)
    reduceDimension(SDAEFeaFile, same_ratio, SDAEFeaFile_reduce)
    reduceDimension(SDAEFeaFile_s, same_ratio, SDAEFeaFile_reduce_s)
