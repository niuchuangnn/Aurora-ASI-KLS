import numpy as np
import h5py
from src.util.paseLabeledFile import parseNL

if __name__ == '__main__':
    labeledFile = '../../Data/type4_b500.txt'
    names, labels = parseNL(labeledFile)
    bboxFile = '../../Data/type4_b500_bbox.hdf5'
    save_bboxFile = '../../Data/type4_b500_whole_bbox.hdf5'
    f = h5py.File(bboxFile, 'r')
    fs = h5py.File(save_bboxFile, 'w')
    for i in range(len(f)):
        im_i = f.get(str(i))
        imgFile = im_i.attrs['imgFile']
        name = im_i.attrs['imgName']
        regions_special = im_i.get('bbox_special')
        regions_common = im_i.get('bbox_common')
        labels_special = im_i.get('labels_special')
        labels_common = im_i.get('labels_common')
        angle_special = regions_special.attrs['angle']
        angle_common = regions_common.attrs['angle']
        label = int(labels[names.index(name)])
        print regions_special.shape

        whole = np.array([[0, 0, 440, 440]])
        if regions_special.shape[0] == 0:
            regions_special = whole
        else:
            regions_special = np.append(regions_special, whole, axis=0)
        # print label
        if labels_special.shape[0] != 0:
            # print labels_special[0]
            if labels_special[0] == int(label):
                print 'true'
        labels_special = np.zeros((regions_special.shape[0], ), dtype='i')
        labels_special.fill(label)
        print regions_special.shape

        group = fs.create_group(str(i))
        group.attrs['imgFile'] = imgFile
        group.attrs['imgName'] = name
        d_special = group.create_dataset('bbox_special', shape=regions_special.shape, dtype='i', data=regions_special)
        d_common = group.create_dataset('bbox_common', shape=regions_common.shape, dtype='i', data=regions_common)
        group.create_dataset('labels_special', shape=labels_special.shape, dtype='i', data=labels_special)
        group.create_dataset('labels_common', shape=labels_common.shape, dtype='i', data=labels_common)
        d_special.attrs['angle'] = angle_special
        d_common.attrs['angle'] = angle_common
        print name, 'bbox saved'
    f.close()
    fs.close()