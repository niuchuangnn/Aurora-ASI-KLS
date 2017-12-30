import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.misc import imread, imsave
from src.experiments.test_segmentation_accuracy import load_mask_mat

classNum = 4
root_path = '../../Data/segmentation_data_v2/'

# mask_type = '.mat'
img_type = '.jpg'
for i in xrange(classNum):
    mask_root_path = root_path + str(i+1) + '_mask/'
    maskImage_saveFolder = root_path + str(i+1) + '_maskImage/'
    if not os.path.exists(maskImage_saveFolder):
        os.mkdir(maskImage_saveFolder)
    files = os.listdir(mask_root_path)
    for fn in files:
        mask_i_path = mask_root_path + fn
        mask = load_mask_mat(mask_i_path)
        imsave(maskImage_saveFolder+fn[:-4]+img_type, mask*255)