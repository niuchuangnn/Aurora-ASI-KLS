import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.misc import imread
from src.experiments.test_segmentation_accuracy import load_mask_mat

classNum = 4
root_path = '../../Data/segmentation_data_v2/'

# mask_type = '.mat'
img_type = '.jpg'
for i in xrange(classNum):
    mask_root_path = root_path + str(i+1) + '_mask/'
    img_root_path = root_path + str(i+1) + '_selected/'
    files = os.listdir(mask_root_path)
    for fn in files:
        mask_i_path = mask_root_path + fn
        mask = load_mask_mat(mask_i_path)
        # mask = sio.loadmat(mask_i_path)['mask']
        # # mask = np.transpose(mask, (1, 0))
        # for x in xrange(mask.shape[0]):
        #     mask[x, :] = mask[x, ::-1]
        # for y in xrange(mask.shape[1]):
        #     mask[:, y] = mask[::-1, y]
        # mask = mask.T
        img_i_path = img_root_path + fn[:-4] + img_type
        im = imread(img_i_path)
        plt.figure(1)
        plt.imshow(mask, cmap='gray')
        plt.figure(2)
        plt.imshow(im, cmap='gray')
        plt.show()
#mask_name = 'N20041112G140201'
#mask_path = root_path +
#mask = sio.loadmat(path)['mask']
#name = mask_path[-20:-4]
#img
#plt.figure(1)
#plt.imshow(mask, cmap='gray')
#plt.show()