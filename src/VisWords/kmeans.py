import numpy as np
from sklearn.cluster import KMeans
import h5py

if __name__ == '__main__':
    siftFeaFile = '../../Data/Features/balance500SIFT.hdf5'
    f = h5py.File(siftFeaFile, 'r')
    for name in f:
        print name
    dataFolder = f.attrs['dataFolder']
    feaSet = f.get('feaSet')
    posSet = f.get('posSet')
    auroraData = f.get('auroraData')

    # classes = ['1', '2', '3', '4']
    classes = ['3', '4']

    Kmeans = KMeans(n_clusters=500, n_jobs=-1)

    saveFolder = '../../Data/Features/'
    saveName = 'SIFTWords.hdf5'
    f_w = h5py.File(saveFolder+saveName, 'a')

    for c in classes:
        feas = feaSet.get(c)
        feas = np.array(feas)
        print 'feas shape: ' + str(feas.shape)
        Kmeans.fit(feas)
        cluster_centers = Kmeans.cluster_centers_
        inertia = Kmeans.inertia_
        cc = f_w.create_group(c)
        cc.attrs['inertia'] = inertia
        cc.create_dataset('words', cluster_centers.shape, 'f', cluster_centers)
    print saveFolder+saveName+' saved'