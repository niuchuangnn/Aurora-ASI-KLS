import skimage.data
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../../selective_search_py')
import argparse
import warnings
import numpy
import skimage.io
import features
import color_space
import selective_search
import src.util.paseLabeledFile as plf
import os
from scipy.misc import imread, imresize, imsave
from skimage.transform import rotate
import h5py
import warnings
warnings.filterwarnings("ignore")

def generate_color_table(R):
    # generate initial color
    colors = numpy.random.randint(0, 255, (len(R), 3))

    # merged-regions are colored same as larger parent
    for region, parent in R.items():
        if not len(parent) == 0:
            colors[region] = colors[parent[0]]

    return colors

def filterOverlap(regions, thresh):
    if regions.shape[0] != 0:
        x1 = regions[:, 0]
        y1 = regions[:, 1]
        h = regions[:, 2]
        w = regions[:, 3]
        x2 = x1 + h
        y2 = y1 + w
        areas = h * w
        order = areas.argsort()[::-1]
    #     scores = dets[:, 4]
    #
    #     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #     order = scores.argsort()[::-1]
    #
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(regions[i, :])
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            ww = np.maximum(0.0, xx2 - xx1 + 1)
            hh = np.maximum(0.0, yy2 - yy1 + 1)
            inter = ww * hh
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
    else:
        keep = []
    return keep

def special_common_local_proposal(paras):
    proposal_minSize = paras['proposal_minSize']
    proposal_maxSize = paras['proposal_maxSize']
    overlap_th = paras['overlap_th']
    train = paras['train']
    is_rotate = paras['is_rotate']
    region_set = selective_search.selective_search_M(paras)
    if train:
        print 'regions_set', region_set[0]
        regions = {}
        for rs in region_set:
            ri = rs[0]
            for rk, rv in ri.iteritems():
                if rk not in regions:
                    regions[rk] = []

                for r in rv:
                    x = r[1][0]
                    y = r[1][1]
                    h = r[1][2] - x
                    w = r[1][3] - y
                    if (w*h < proposal_maxSize) and (w*h > proposal_minSize):
                        regions[rk].append([x, y, h, w])
        keep = {}
        for kk, vv in regions.iteritems():
            regions_n = np.array(vv)
            keep[kk] = filterOverlap(regions_n, overlap_th)
    else:
        regions = []
        for rs in region_set:
            # print region_set
            ri = rs[0]
            if is_rotate:
                RL = rs[1]
                angle = rs[2]
            for r in ri:
                if is_rotate:
                    sub_regions = r[-1]
                    im_psudo = np.zeros((440, 440), dtype='i')
                    # print sub_regions
                    for l in sub_regions:
                        # print 'l', l
                        im_psudo[np.where(RL==l)] = 255
                    # print 'before size: ', len(im_psudo[np.where(im_psudo==255)])
                    im_rotate = rotate(im_psudo, angle, preserve_range=True)
                    I, J = np.where(im_rotate > 100)
                    # print 'after size: ', len(im_rotate[np.where(im_rotate>100)])
                    # print I, J
                    [x, y, x2, y2] = [min(I), min(J), max(I), max(J)]
                    h = x2 - x
                    w = y2 - y
                else:
                    x = r[1][0]
                    y = r[1][1]
                    h = r[1][2] - x
                    w = r[1][3] - y
                if (w * h < proposal_maxSize) and (w * h > proposal_minSize):
                    regions.append([x, y, h, w])
        regions_n = np.array(regions)
        keep = filterOverlap(regions_n, overlap_th)
    # keep = list(keep)
    # regions_n[:, 2] = regions_n[:, 2] - regions_n[:, 0]
    # regions_n[:, 3] = regions_n[:, 3] - regions_n[:, 1]
    if is_rotate:
        return keep, angle
    else:
        return keep

def save_hierachecalRegionsProcess(paras):
    imName = paras['imgFile'][-20:-4]
    eraseMap = paras['eraseMap']
    returnRegionLabels = paras['returnRegionLabels']
    feature_masks = paras['feature_masks']
    hierachecalProcessPath = paras['hierachecalProcessPath']
    if 0 in returnRegionLabels:
        cs = '_C'
    else:
        cs = '_S'
    imName += cs
    paras['thresh'] = 0
    (R, F, L, L_regions, eraseLabels, angle) = selective_search.hierarchical_segmentation_M(paras, feature_masks)
    print('result filename: %s_[0000-%04d].png' % (imName, len(F) - 1))

    # suppress warning when saving result images
    warnings.filterwarnings("ignore", category=UserWarning)

    colors = generate_color_table(R)
    for depth, label in enumerate(F):
        result = colors[label]
        result = (result * alpha + img * (1. - alpha)).astype(numpy.uint8)
        fn = "%s_%04d.png" % (imName, depth)
        result[numpy.where(eraseMap==1)] = 0
        skimage.io.imsave(hierachecalProcessPath+fn, result)
        if depth == 0:
            specialMap = result.astype(numpy.uint8)
            for el in eraseLabels:
                specialMap[numpy.where(F[0] == el)] = 0
            skimage.io.imsave(hierachecalProcessPath + imName + '.png', specialMap)
            specialMap_rotate = rotate(specialMap, angle, preserve_range=True).astype(numpy.uint8)
            skimage.io.imsave(hierachecalProcessPath + imName + '_rotate.png', specialMap_rotate)
        sys.stdout.write('.')
        sys.stdout.flush()

    print('\n')
    return 0

def generateSpecialCommonBbox(labelFile, savePath, dataFolder, imgType, paras):
    names, labels = plf.parseNL(labelFile)
    fs = h5py.File(savePath, 'w')
    for i in range(len(names)):
        name = names[i]
        label = labels[i]
        imgFile = dataFolder + name + imgType
        paras['imgFile'] = imgFile
        im = skimage.io.imread(imgFile)
        if len(im.shape) == 2:
            img = skimage.color.gray2rgb(im)
        paras['im'] = img
        paras['img'] = img
        paras['specialType'] = int(label) - 1  # 0: arc, 1: drapery, 2: radial, 3: hot-spot

        paras['th'] = 0.45
        paras['returnRegionLabels'] = [1, 2]  # 0: special, 1: rest, 2: common
        regions_special, angle_special = special_common_local_proposal(paras)
        regions_special = np.array(regions_special)

        paras['th'] = 0.25
        paras['returnRegionLabels'] = [0, 1]  # 0: special, 1: rest, 2: common
        regions_common, angle_common = special_common_local_proposal(paras)
        regions_common = np.array(regions_common)

        labels_special = np.zeros((regions_special.shape[0],), dtype='i')
        labels_special.fill(label)
        labels_common = np.zeros((regions_common.shape[0],), dtype='i')

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
    fs.close()
    return 0

if __name__=="__main__":
    k = 100
    feature_masks = [1, 1, 1, 1]  # ['size', 'color', 'texture', 'fill']
    out_prefix = ''
    alpha = 0.5  # alpha value for compositing result image with input image

    paras = {}
    paras['k'] = 100
    paras['minSize'] = 100
    paras['patchSize'] = np.array([16, 16])
    paras['region_patch_ratio'] = 0.1
    paras['sigma'] = 0.5
    paras['alpha'] = 0.6
    paras['overlap_th'] = 0.5
    paras['types'] = ['arc', 'drapery', 'radial', 'hot_spot']
    paras['lbp_wordsFile_s1'] = '../../Data/Features/type4_LBPWords_s1_s16_b300_w500.hdf5'
    paras['lbp_wordsFile_s2'] = '../../Data/Features/type4_LBPWords_s2_s16_b300_w500.hdf5'
    paras['lbp_wordsFile_s3'] = '../../Data/Features/type4_LBPWords_s3_s16_b300_w500.hdf5'
    paras['lbp_wordsFile_s4'] = '../../Data/Features/type4_LBPWords_s4_s16_b300_w500.hdf5'

    paras['sift_wordsFile_s1'] = '../../Data/Features/type4_SIFTWords_s1_s16_300_300_300_300.hdf5'
    paras['sift_wordsFile_s2'] = '../../Data/Features/type4_SIFTWords_s2_s16_300_300_300_300.hdf5'
    paras['sift_wordsFile_s3'] = '../../Data/Features/type4_SIFTWords_s3_s16_300_300_300_300.hdf5'
    paras['sift_wordsFile_s4'] = '../../Data/Features/type4_SIFTWords_s4_s16_300_300_300_300.hdf5'

    eraseMapPath = '../../Data/eraseMap.bmp'
    if not os.path.exists(eraseMapPath):
        imSize = 440
        eraseMap = np.zeros((imSize, imSize))
        radius = imSize / 2
        centers = np.array([219.5, 219.5])
        for i in range(440):
            for j in range(440):
                if np.linalg.norm(np.array([i, j]) - centers) > 220:
                    eraseMap[i, j] = 1
        imsave(eraseMapPath, eraseMap)
    else:
        eraseMap = imread(eraseMapPath) / 255

    paras['eraseMap'] = eraseMap
    paras['sizeRange'] = (16, 16)
    paras['imgSize'] = (440, 440)
    paras['nk'] = 1
    resolution = 1
    gridSize = np.array([resolution, resolution])
    paras['resolution'] = resolution
    paras['gridSize'] = gridSize

    sdaePara = {}
    sdaePara['weight_d'] = '../../Data/autoEncoder/layer_diff_mean_s16_final.caffemodel'
    sdaePara['weight_s'] = '../../Data/autoEncoder/layer_same_mean_s16_final.caffemodel'
    sdaePara['net'] = '../../Data/autoEncoder/test_net.prototxt'
    sdaePara['meanFile'] = '../../Data/patchData_mean_s16.txt'
    sdaePara['patchMean'] = False
    # layerNeuronNum = [28 * 28, 2000, 1000, 500, 128]
    layerNeuronNum = [16 * 16, 1000, 1000, 500, 64]
    sdaePara['layerNeuronNum'] = layerNeuronNum

    paras['sdaePara'] = sdaePara

    paras['feaType'] = 'LBP'
    paras['withIntensity'] = False
    paras['diffResolution'] = False
    paras['mk'] = None
    paras['thresh'] = 0
    paras['isSave'] = False
    paras['feature_masks'] = feature_masks
    color_space = ['rgb']
    ks = [40]
    paras['ks'] = ks
    paras['color_spaces'] = color_space

    proposal_minSize = 100 * 100
    proposal_maxSize = 440 * 440
    paras['proposal_minSize'] = proposal_minSize
    paras['proposal_maxSize'] = proposal_maxSize
    paras['hierachecalProcessPath'] = '../../Data/Results/hierachicalProcess/'
    paras['train'] = False
    paras['is_rotate'] = True
    paras['eraseRegionLabels'] = [2]  # 0: special, 1: rest, 2: common

    # labelFile = '../../Data/type4_300_300_300_300.txt'
    labelFile = '../../Data/type4_b500.txt'
    dataFolder = '../../Data/labeled2003_38044/'
    # savePath = '../../Data/type4_b300_bbox.hdf5'
    savePath = '../../Data/type4_b500_SR_100_440_bbox.hdf5'
    imgType = '.bmp'
    generateSpecialCommonBbox(labelFile, savePath, dataFolder, imgType, paras)

    paras['specialType'] = 3  # 0: arc, 1: drapery, 2: radial, 3: hot-spot
    paras['returnRegionLabels'] = [1, 2]  # 0: special, 1: rest, 2: common
    paras['th'] = 0.45

    figSaveFolder = '../../Data/Results/training_samples/'
    imgFile = '../../Data/labeled2003_38044/N20031221G103321.bmp'
    name = imgFile[-20:-4]
    if 0 in paras['returnRegionLabels']:
        c_s = '_common'
    else:
        c_s = '_special'
    name += c_s
    paras['imgFile'] = imgFile
    im = skimage.io.imread(imgFile)
    if len(im.shape) == 2:
        img = skimage.color.gray2rgb(im)
    paras['im'] = img
    paras['img'] = img
    save_hierachecalRegionsProcess(paras)
    if paras['is_rotate']:
        regions, angle = special_common_local_proposal(paras)
    else:
        regions = special_common_local_proposal(paras)
    # regions = list(regions)
    print np.array(regions)
    if paras['train']:
        for k, v in regions.iteritems():
            plf.showGrid(img, v)
            plt.title(k)
        plt.show()
    else:
        if paras['is_rotate']:
            img = rotate(img, angle)
        plf.showGrid(img, regions)
        plt.savefig(figSaveFolder+name+'.png')
        plt.show()

    # im = skimage.io.imread(imgFile)
    # if len(im.shape) == 2:
    #     img = skimage.color.gray2rgb(im)

    # (R, F, L, L_regions) = selective_search.hierarchical_segmentation(img, k, feature_masks, eraseMap=None)
    # print('result filename: %s_[0000-%04d].png' % (out_prefix, len(F) - 1))
    #
    # # suppress warning when saving result images
    # warnings.filterwarnings("ignore", category=UserWarning)

    # colors = generate_color_table(R)
    # for depth, label in enumerate(F):
    #     result = colors[label]
    #     result = (result * alpha + img * (1. - alpha)).astype(numpy.uint8)
    #     fn = "%s_%04d.png" % (out_prefix, depth)
    #     skimage.io.imsave(fn, result)
    #     sys.stdout.write('.')
    #     sys.stdout.flush()
    #
    # print('\n')
    #
    # erase_map = np.zeros((440, 440))
    # centers = np.array([219.5, 219.5])
    # for i in range(440):
    #     for j in range(440):
    #         if np.linalg.norm(np.array([i, j]) - centers) > 220+5:
    #             erase_map[i, j] = 1
    # (R, F, L, L_regions) = selective_search.hierarchical_segmentation(img, k, feature_masks, eraseMap=erase_map)
    #
    # # region_img = F[0]
    # # for i in range(60,80):
    # #     grid = [L[i]]
    # #     sub_regions = L_regions[i]
    # #     im_psudo = np.zeros((440, 440))
    # #     for l in sub_regions:
    # #         im_psudo[np.where(region_img == l)] = 255
    # #     plf.showGrid(im_psudo, grid)
    # # plt.show()
    #
    # out_prefix = 'erase'
    # print('result filename: %s_[0000-%04d].png' % (out_prefix, len(F) - 1))
    #
    # # suppress warning when saving result images
    # warnings.filterwarnings("ignore", category=UserWarning)
    #
    # colors = generate_color_table(R)
    # for depth, label in enumerate(F):
    #     result = colors[label]
    #     result = (result * alpha + img * (1. - alpha)).astype(numpy.uint8)
    #     fn = "%s_%04d.png" % (out_prefix, depth)
    #     skimage.io.imsave(fn, result)
    #     sys.stdout.write('.')
    #     sys.stdout.flush()
    #
    # print('\n')


    # region_labels = []
    # for i in range(len(region_set)):
    #     region_labels.append(region_set[i][-1])
    #     region_set[i] = region_set[i][0]

    # grids = [x[1] for x in regions]


    # regions = []
    # for rs in region_set:
    #     ri = rs[0]
    #     I = rs[1]
    #     eraseLabels = set(list(I[numpy.where(eraseMap == 1)].flatten()))
    #     regions = []
    #     for r in ri:
    #         exist_eraseLabels = [l for l in eraseLabels if l in r[2]]
    #         if len(exist_eraseLabels) == 0:
    #             grid_axes = r[1]
    #             h = grid_axes[2] - grid_axes[0]
    #             w = grid_axes[3] - grid_axes[1]
    #             if (h*w >= proposal_minSize) and (h*w <= proposal_maxSize):
    #                 regions.append(r)

        # regions = [x[0] for x in region_set]
        # region_imgs = [x[-1] for x in region_set]

        # proposals = sorted(regions[0])
        # region_img = region_imgs[0]
        # proposals = regions
        # region_img = I
        #
        # for i in range(len(regions)):
        #     cornors = proposals[i][1]
        #     grid = [[cornors[0], cornors[1], cornors[2] - cornors[0] + 1, cornors[3] - cornors[1] + 1]]
        #     sub_regions = proposals[i][-1]
        #     im_psudo = np.zeros((440, 440))
        #     for l in sub_regions:
        #         im_psudo[np.where(region_img==l)] = im[np.where(region_img==l)]
        #     plf.showGrid(im_psudo, grid)

        # plt.figure(2)

        # plt.imshow(map, cmap='gray')
        # plt.show()
    # pass