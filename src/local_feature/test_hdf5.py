import h5py
import numpy as np

f = h5py.File('/home/ljm/NiuChuang/KLSA-auroral-images/Data/Features/type4_SIFTWords_h1_s16_600_300_300_300.hdf5', 'r')
for i in f:
    print f.get(i + '/words').shape

# f = h5py.File('test.hdf5', 'w')
# dset = f.create_dataset('mydataset', (100,), dtype='i')
# print dset.shape
# print dset.dtype
#
# dset[...] = np.arange(100)
# print dset[0],dset[10]
# print dset[0:100:10]
#
# print dset.name
# print f.name
#
# grp = f.create_group('subgroup')
# dset2 = grp.create_dataset('another_dataset', (50,), dtype='f')
# print dset2.name
#
# dset3 = f.create_dataset('subgroup/dataset_three', (10,), dtype='i')
# print dset3.name
#
# dataset_three = f['subgroup/dataset_three']
#
# for name in f:
#     print name
#
# print 'mydataset' in f
# print 'somethingelse' in f
# print 'subgroup/another_dataset' in f
#
# def printname(name):
#     print name
# f.visit(printname)
#
# dset.attrs['temperature'] = 99.5
# print dset.attrs['temperature']
#
# print 'temperature' in dset.attrs
#
# dset.attrs['s'] = 'sdge'
# print dset.attrs['s']
#
# print 's' in dset.attrs
#
# strList = ['sdf', 'dfg4', 'bght']
# asciiList = [s.encode('ascii', 'ignore') for s in strList]
# f.create_dataset('strdataset', (len(asciiList),1), 'S10', asciiList)
# f.close()
# pass