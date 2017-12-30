import h5py
import numpy as np
import sys
sys.path.insert(0, '../')
import utils.show_kls as showkls
from scipy.misc import imread

def get_aurora_klsdb(bbox_path, dataFolder, imgType):
    f = h5py.File(bbox_path, 'r')

    klsdb = []
    for i in range(len(f)):
        imgName = f.get(str(i)).attrs['imgName']
        bbox_special = np.array(f.get(str(i) + '/bbox_special'))
        bbox_common = np.array(f.get(str(i) + '/bbox_common'))

        angle_special = f.get(str(i) + '/bbox_special').attrs['angle']
        angle_common = f.get(str(i) + '/bbox_common').attrs['angle']
        label_special = np.array(f.get(str(i) + '/labels_special'))
        label_common = np.array(f.get(str(i) + '/labels_common'))

        if bbox_special.shape[0] != 0:
            imi_s = {}
            # convert [y1, x1, h, w] to [y1, x1, y2, x2]
            bbox_special[:, 2] = bbox_special[:, 0] + bbox_special[:, 2]
            bbox_special[:, 3] = bbox_special[:, 1] + bbox_special[:, 3]
            # convert [y1, x1, y2, x2] to [x1, y1, x2, y2]
            bbox_special[:, [0, 1]] = bbox_special[:, [1, 0]]
            bbox_special[:, [2, 3]] = bbox_special[:, [3, 2]]
            imi_s['image'] = dataFolder + imgName + imgType
            imi_s['bbox'] = bbox_special
            imi_s['angle'] = angle_special
            imi_s['gt_classes'] = label_special
            klsdb.append(imi_s)
        if bbox_common.shape[0] != 0:
            imi_c = {}
            # convert [y1, x1, h, w] to [y1, x1, y2, x2]
            bbox_common[:, 2] = bbox_common[:, 0] + bbox_common[:, 2]
            bbox_common[:, 3] = bbox_common[:, 1] + bbox_common[:, 3]
            # convert [y1, x1, y2, x2] to [x1, y1, x2, y2]
            bbox_common[:, [0, 1]] = bbox_common[:, [1, 0]]
            bbox_common[:, [2, 3]] = bbox_common[:, [3, 2]]
            imi_c['image'] = dataFolder + imgName + imgType
            imi_c['bbox'] = bbox_common
            imi_c['angle'] = angle_common
            imi_c['gt_classes'] = label_common
            klsdb.append(imi_c)

    return klsdb

def calculate_klsdb_mean(klsdb):
    im_num = len(klsdb)
    means = 0.
    for i in xrange(im_num):
        imi = klsdb[i]
        imgFile = imi['image']
        im = imread(imgFile)
        means += im.mean()
    mean = means / im_num
    return mean

if __name__ == '__main__':
    # bbox_path = '/home/niuchuang/PycharmProjects/KLSA-auroral-images/Data/type4_b300_bbox.hdf5'
    bbox_path = '/home/amax/NiuChuang/KLSA-auroral-images/Data/type4_b500_SR_100_440_bbox.hdf5'
    data_folder = '/home/amax/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/'
    imgType = '.bmp'
    klsdb = get_aurora_klsdb(bbox_path, data_folder, imgType)
    mean = calculate_klsdb_mean(klsdb)
    print mean
    showkls.show_kls(klsdb)
    pass