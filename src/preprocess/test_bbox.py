import h5py
import numpy as np
import src.util.paseLabeledFile as plf
from scipy.misc import imread, imresize, imsave
from skimage.transform import rotate
import matplotlib.pyplot as plt

if __name__ == '__main__':
    bbox_file = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/type4_b500_SR_100_400_bbox.hdf5'
    f = h5py.File(bbox_file, 'r')

    for i in range(1800, len(f)):
        imgFile = f.get(str(i)).attrs['imgFile']
        imgName = f.get(str(i)).attrs['imgName']
        bbox_special = f.get(str(i)+'/bbox_special')
        bbox_common = f.get(str(i)+'/bbox_common')
        angle_special = f.get(str(i)+'/bbox_special').attrs['angle']
        angle_common = f.get(str(i)+'/bbox_common').attrs['angle']
        labels_special = f.get(str(i)+'/labels_special')
        labels_common = f.get(str(i)+'/labels_common')
        # print labels_common.shape[0]
        # print bbox_common.shape[0]
        #
        # b_s = np.array(bbox_special)
        # b_c = np.array(bbox_common)
        #
        # l_s = np.array(labels_special).reshape(b_s.shape[0], 1)
        # l_c = np.array(labels_common).reshape(b_c.shape[0], 1)
        # print l_s.shape
        #
        # bb = np.zeros((0, 4), dtype='i')
        # ll = np.zeros((0, 1), dtype='i')
        # if b_s.shape[0] != 0:
        #     bb = np.vstack([bb, b_s])
        #     ll = np.vstack([ll, l_s])
        # if b_c.shape[0] != 0:
        #     ll = np.vstack([ll, l_c])
        #     bb = np.vstack([bb, b_c])
        # print b_s
        # print b_c
        # print bb
        # print l_s
        # print l_c
        # print ll
        im = imread(imgFile)
        im_s = rotate(im, angle_special)
        bbox_s = list(bbox_special)
        bbox_c = list(bbox_common)
        plf.showGrid(im_s, bbox_s)
        print imgName
        plt.title(imgName+'_special')
        im_c = rotate(im, angle_common)
        plf.showGrid(im_c, bbox_c)
        plt.title(imgName+'_common')

        plt.show()