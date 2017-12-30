import sys
sys.path.insert(0, '../../selective_search_py')
sys.path.insert(0, '../../fast-rcnn/lib')
sys.path.insert(0, '../../fast-rcnn/caffe-fast-rcnn/python')
import caffe
from fast_rcnn.test_kls import im_detect
import selective_search as ss
import skimage.io
import os
from scipy.misc import imread, imsave
from src.preprocess.proposal_ss import filterOverlap
import src.util.paseLabeledFile as plf
import segment
import numpy as np
import matplotlib.pyplot as plt
from src.local_feature.adaptiveThreshold import calculateThreshold
import time
from skimage import feature
import src.local_feature.dsift as dsift
import h5py

def normalizeVecs(vecs):
    len_vecs = np.sqrt(np.sum(vecs ** 2, axis=1))
    vecs = vecs / len_vecs.reshape((len_vecs.size, 1))
    return vecs

def isWithinCircle(grid, centers, radius):
    flag = True
    [h, w] = [grid[2], grid[3]]
    upperLeft = np.array([grid[0], grid[1]])
    upperRight = np.array([grid[0], grid[1] + w - 1])
    lowerLeft = np.array([grid[0] + h - 1, grid[1]])
    lowerRight = np.array([grid[0] + h - 1, grid[1] + w - 1])
    coordinates = [upperLeft, upperRight, lowerLeft, lowerRight]
    for c in coordinates:
        if np.linalg.norm(c - centers) > radius:
            flag = False
            break
    return flag

def generateGridPatchData(im, gridList):

    gridPatchData = []
    for grid in gridList:
        if im.ndim == 2:
            patch = im[int(grid[0]):int(grid[0]+grid[2]), int(grid[1]):int(grid[1]+grid[3])].copy()     # grid format: [y, x, h, w] (y: row, x: column)
        if im.ndim == 3:
            patch = im[grid[0]:(grid[0] + grid[2]), grid[1]:(grid[1] + grid[3]), :].copy()  # grid format: [y, x, h, w]
        gridPatchData.append(patch)
    return gridPatchData, gridList

def generate_subRegions(img, patchSize, region_patch_ratio, eraseMap, k, minSize, sigma, thresh=0, radius=220, centers = np.array([219.5, 219.5])):

    im = img[:, :, 0]

    F0, n_region = segment.segment_label(img, sigma, k, minSize)

    eraseLabels = set(list(F0[np.where(eraseMap == 1)].flatten()))

    region_patch_list = [[] for i in range(n_region)]

    for l in range(n_region):
        if l in eraseLabels:
            region_patch_list[l] = []
        else:
            region_patch_centers = list(np.argwhere(F0 == l))
            if len(region_patch_centers) == 0:
                continue
            region_values = im[np.where(F0 == l)]
            region_mean = region_values.mean()
            # print thresh
            if region_mean < thresh:
                region_patch_list[l] = []
            else:
                hw = patchSize / 2
                region_patch_gride = np.zeros((len(region_patch_centers), 4))
                region_patch_gride[:, :2] = np.array(region_patch_centers) - hw
                region_patch_gride[:, 2:] = patchSize
                patch_list = list(region_patch_gride)
                for ii in range(len(region_patch_centers)):
                    ll = patch_list[ii]
                    if np.random.rand(1, )[0] < region_patch_ratio:
                        if isWithinCircle(ll, centers, radius):
                            region_patch_list[l].append(ll)
    return F0, region_patch_list, eraseLabels

def calImgLBPFeatures(im, gridList, norm=True):
    P1 = 8
    P2 = 16
    P3 = 24
    R1 = 1
    R2 = 2
    R3 = 3

    gridPatchData, positions = generateGridPatchData(im, gridList)
    LBPFeaDim = 10+18+26
    feaVecs = np.zeros((len(gridPatchData), LBPFeaDim))
    for i in range(len(gridPatchData)):
        LBP_img_R1P8 = feature.local_binary_pattern(gridPatchData[i], P1, R1, method='uniform')
        LBP_img_R2P16 = feature.local_binary_pattern(gridPatchData[i], P2, R2, method='uniform')
        LBP_img_R3P24 = feature.local_binary_pattern(gridPatchData[i], P3, R3, method='uniform')

        lbp_bin_num_R1P8 = P1 + 2
        lbp_hist_R1P8, lbp_bins_R1P8 = np.histogram(LBP_img_R1P8.flatten(), bins=range(lbp_bin_num_R1P8 + 1))

        lbp_bin_num_R2P16 = P2 + 2
        lbp_hist_R2P16, lbp_bins_R2P16 = np.histogram(LBP_img_R2P16.flatten(), bins=range(lbp_bin_num_R2P16 + 1))

        lbp_bin_num_R3P24 = P3 + 2
        lbp_hist_R3P24, lbp_bins_R3P24 = np.histogram(LBP_img_R3P24.flatten(), bins=range(lbp_bin_num_R3P24 + 1))

        feaVec = np.array(list(lbp_hist_R1P8) + list(lbp_hist_R2P16) + list(lbp_hist_R3P24))

        feaVecs[i, :] = feaVec
    if norm:
        feaVecs = normalizeVecs(feaVecs)

    return feaVecs, np.array(positions)

def calImgDSift(im, gridList):
    siftFeaDim = 128
    patches, positions = generateGridPatchData(im, gridList)
    feaVecs = np.zeros((len(patches), siftFeaDim))
    for i in range(len(patches)):
        patchSize = int(positions[i][-1])
        extractor = dsift.SingleSiftExtractor(patchSize)
        feaVec = extractor.process_image(patches[i])
        feaVecs[i, :] = feaVec

    return feaVecs, np.array(positions)

def histogramOfPatch(patch_arr, num_bins):
    hist, bins = np.histogram(patch_arr.flatten(), bins=range(0, 256 + 1, 256/num_bins))
    return hist, bins


def calImgHisFeatures(im, gridList, HisFeaDim=64, norm=True):
    im = (im*255).astype(np.uint8)
    gridPatchData, positions = generateGridPatchData(im, gridList)
    feaVecs = np.zeros((len(gridPatchData), HisFeaDim))
    for i in range(len(gridPatchData)):
        feaVec, _ = histogramOfPatch(gridPatchData[i], HisFeaDim)
        feaVecs[i, :] = feaVec
    if norm:
        feaVecs = normalizeVecs(feaVecs)
    return feaVecs, np.array(positions)

def genImgLocalFeas(im, feaType, gridList, HisFeaDim=64, norm=True):

    if feaType == 'SIFT':
        feaVectors, posVectors = calImgDSift(im, gridList)

    if feaType == 'LBP':
        feaVectors, posVectors = calImgLBPFeatures(im, gridList, norm=norm)

    if feaType == 'His':
        feaVectors, posVectors = calImgHisFeatures(im, gridList, HisFeaDim=HisFeaDim, norm=norm)

    return feaVectors, posVectors

def calPatchLabels2(wordsFile, feaVectors, k=11, two_classes=['1', '2'], L=None):
    fw = h5py.File(wordsFile, 'r')
    w1 = fw.get(two_classes[0] + '/words')
    w2 = fw.get(two_classes[1] + '/words')
    w1 = np.array(w1)
    w2 = np.array(w2)
    num_words = w1.shape[0]
    patch_num = feaVectors.shape[0]
    dis1 = np.zeros((patch_num, num_words))
    dis2 = np.zeros((patch_num, num_words))
    label1 = np.zeros(num_words)  # class1: 0, class2: 1, common: 2
    label2 = np.ones(num_words)

    for v in range(patch_num):
        dis1[v, :] = np.linalg.norm(w1 - feaVectors[v], axis=1)
        dis2[v, :] = np.linalg.norm(w2 - feaVectors[v], axis=1)
    dis = np.append(dis1, dis2, axis=1)

    if (L is not None) and (L != 0):
        common_vec_name = 'common_vectors' + str(L)
    else:
        common_vec_name = 'common_vectors'

    w1_common_idx = np.array(fw.get(common_vec_name+'/common_vec_' + two_classes[0]))
    w2_common_idx = np.array(fw.get(common_vec_name+'/common_vec_' + two_classes[1]))

    # print w1_common_idx
    w1_common_list = list(w1_common_idx.reshape(len(w1_common_idx)))
    w2_common_list = list(w2_common_idx.reshape(len(w2_common_idx)))
    label1[w1_common_list] = 2
    label2[w2_common_list] = 2
    labels = np.append(label1, label2)

    dis_sort_idx = np.argsort(dis, axis=1)
    dis_min_idx_k = dis_sort_idx[:, :k]

    patchLabels = np.zeros((patch_num, k))
    for i in range(patch_num):
        patchLabels[i, :] = labels[list(dis_min_idx_k[i, :])]

    return patchLabels

def region_special_map(paras):
    img = paras['img']
    k = paras['k']
    minSize = paras['minSize']
    patchSize = paras['patchSize']
    region_patch_ratio = paras['region_patch_ratio']
    sigma = paras['sigma']
    th = paras['th']
    nk = paras['nk']
    eraseMap = paras['eraseMap']
    feaType = paras['feaType']
    types = paras['types']
    specialType = paras['specialType']
    thresh = paras['thresh']
    L = paras['L']
    im_norm = paras['im_norm']

    if feaType == 'His':
        his_wordsFile_s1 = paras['his_wordsFile_s1']
        his_wordsFile_s2 = paras['his_wordsFile_s2']
        his_wordsFile_s3 = paras['his_wordsFile_s3']
        his_wordsFile_s4 = paras['his_wordsFile_s4']
        wordsFile_s = [his_wordsFile_s1, his_wordsFile_s2, his_wordsFile_s3, his_wordsFile_s4]

    if feaType == 'LBP':
        lbp_wordsFile_s1 = paras['lbp_wordsFile_s1']
        lbp_wordsFile_s2 = paras['lbp_wordsFile_s2']
        lbp_wordsFile_s3 = paras['lbp_wordsFile_s3']
        lbp_wordsFile_s4 = paras['lbp_wordsFile_s4']
        wordsFile_s = [lbp_wordsFile_s1, lbp_wordsFile_s2, lbp_wordsFile_s3, lbp_wordsFile_s4]

    if feaType == 'SIFT':
        sift_wordsFile_s1 = paras['sift_wordsFile_s1']
        sift_wordsFile_s2 = paras['sift_wordsFile_s2']
        sift_wordsFile_s3 = paras['sift_wordsFile_s3']
        sift_wordsFile_s4 = paras['sift_wordsFile_s4']
        wordsFile_s = [sift_wordsFile_s1, sift_wordsFile_s2, sift_wordsFile_s3, sift_wordsFile_s4]

    F0, region_patch_list, eraseLabels = generate_subRegions(img, patchSize, region_patch_ratio, eraseMap, k, minSize, sigma, thresh=thresh)
    maps2by2 = {}
    region_labels = {}
    for ri in range(len(region_patch_list)):
        r = region_patch_list[ri]
        if len(r) != 0:
            feaVectors, posVectors = genImgLocalFeas(im_norm, feaType, gridList=r)
            labels = {}
            w = wordsFile_s[specialType]

            labelVec = calPatchLabels2(w, feaVectors, k=nk, two_classes=['1', '2'], L=L)
            name_s = types[specialType] + '_rest'
            labels[name_s] = labelVec

            for k, v in labels.iteritems():
                v = list(v.flatten())
                if k not in maps2by2:
                    maps2by2[k] = np.zeros((3, F0.shape[0], F0.shape[1]))
                    region_labels[k] = [[], [], []]
                c1 = float(v.count(0)) / float(len(v))
                c2 = float(v.count(1)) / float(len(v))
                cc = float(v.count(2)) / float(len(v))
                cs = np.array([c1, c2, cc])
                cs[np.where(cs < th)] = 0

                maps2by2[k][0][np.where(F0 == ri)] = cs[0]
                maps2by2[k][1][np.where(F0 == ri)] = cs[1]
                maps2by2[k][2][np.where(F0 == ri)] = cs[2]
                if cs[0] > 0:
                    region_labels[k][0].append(ri)
                if cs[1] > 0:
                    region_labels[k][1].append(ri)
                if cs[2] > 0:
                    region_labels[k][2].append(ri)

    return maps2by2, region_labels.values()[0][2], F0

def mergePatchAndRegion(classHeatMaps, categoryHeatMaps, labels, th):
    classHeatMap = classHeatMaps[:, :, labels+1]
    categoryHeatMap = categoryHeatMaps.values()[0][0]
    mergeRusults = (classHeatMap + categoryHeatMap) / 2
    mergeRusults[np.where(mergeRusults > th)] = 1
    mergeRusults[np.where(mergeRusults <= th)] = 0
    return mergeRusults, categoryHeatMap, classHeatMap

def mapsToLabels(classHeatMaps, detection_mask=None):
    regionHeatSizes = np.zeros((classHeatMaps.shape[2]-1, ))
    for i in xrange(1, classHeatMaps.shape[2]):
        map_i = classHeatMaps[:, :, i]
        if detection_mask is not None:
            map_i *= detection_mask
        regionHeatSizes[i-1] = len(list(map_i[np.where(map_i != 0)].flatten()))
    label = regionHeatSizes.argmax()
    return label

def regionSetToBoxes(regionsSet, overlapThresh, sizeRange=[0, 440*440], isVisualize=False):
    bboxList = []
    for region in regionsSet:
        for region_i in region[0]:
            r = region_i[1]
            y = r[0]
            x = r[1]
            h = r[2] - y
            w = r[3] - x
            if (w * h < sizeRange[1]) and (w * h > sizeRange[0]):
                bboxList.append([y, x, h, w])
                if isVisualize:
                    regionMap = region[1]
                    regionLabels = region_i[2]
                    pseudoMap = np.zeros((440, 440))
                    for label in regionLabels:
                        pseudoMap[np.where(regionMap==label)] = 255
                    plf.showGrid(pseudoMap, [[y,x,h,w]])
                    plt.show()
    bboxes = np.array(bboxList)
    keepBoxes = filterOverlap(bboxes, overlapThresh)
    keepBoxes = np.array(keepBoxes)
    # convert [y1, x1, h, w] to [y1, x1, y2, x2]
    keepBoxes[:, 2] = keepBoxes[:, 0] + keepBoxes[:, 2]
    keepBoxes[:, 3] = keepBoxes[:, 1] + keepBoxes[:, 3]
    # convert [y1, x1, y2, x2] to [x1, y1, x2, y2]
    keepBoxes[:, [0, 1]] = keepBoxes[:, [1, 0]]
    keepBoxes[:, [2, 3]] = keepBoxes[:, [3, 2]]
    return keepBoxes

def generateRegionClassHeatMap(scores, boxes, th, imageShape=[440, 440]):
    region_class_heatMap = np.zeros((imageShape[0], imageShape[1], scores.shape[1]))
    heatMap_plusNumber = np.zeros((imageShape[0], imageShape[1], scores.shape[1]))
    for i in xrange(boxes.shape[0]):
        box = boxes[i, :]
        score = scores[i, :]
        label = score.argmax()
        label_score = score[label]
        if label_score > th:
            region_class_heatMap[box[1]:box[3], box[0]:box[2], label] += label_score
            heatMap_plusNumber[box[1]:box[3], box[0]:box[2], label] += 1
    heatMap_plusNumber[np.where(heatMap_plusNumber==0)] = 1
    region_class_heatMap = region_class_heatMap / heatMap_plusNumber
    return region_class_heatMap

def ClsKLSCoarseLoc(imgFile, paras):
    im = imread(imgFile)
    im_norm = np.array(im, dtype='f') / 255
    if len(im.shape) == 2:
        img = skimage.color.gray2rgb(im)
    paras['img'] = img
    paras['im_norm'] = im_norm
    paras['im'] = im
    color_space = paras['color_space']
    ks = paras['ks']
    overlapThresh = paras['overlapThresh']
    regionSizeRange = paras['regionSizeRange']
    net = paras['net']
    scoreThresh = paras['scoreThresh']
    paras['sizeRange'] = (patchSize, patchSize)
    paras['patchSize'] = np.array([patchSize, patchSize])
    paras['feaType'] = feaType
    if L == 0:
        paras['L'] = None
    else:
        paras['L'] = L

    region_set = ss.selective_search(img, color_spaces=color_space, ks=ks,
                                     feature_masks=feature_masks, eraseMap=eraseMap)
    boxes = regionSetToBoxes(region_set, overlapThresh, sizeRange=regionSizeRange, isVisualize=False)
    scores, boxes = im_detect(net, im, boxes)
    classHeatMap = generateRegionClassHeatMap(scores, boxes, scoreThresh)
    label = mapsToLabels(classHeatMap)
    return label, classHeatMap

def KLSLoc(label, classHeatMap, paras):
    paras['specialType'] = label
    paras['thresh'] = calculateThreshold(imgFile)
    maps3, common_labels, F0 = region_special_map(paras)
    kls, categoryMap, classMap = mergePatchAndRegion(classHeatMap, maps3, label, 0.5)
    return kls

if __name__ == '__main__':
    paras = {}
    # parameters of the selective search algorithm
    paras['k'] = 60
    paras['minSize'] = 50
    paras['sigma'] = 0.5
    paras['alpha'] = 0.6
    feature_masks = [1, 1, 1, 1]
    paras['color_space'] = ['rgb']
    # paras['ks'] = [30, 50, 100, 150, 200, 250, 300]
    paras['ks'] = [30, 100, 200, 300]

    paras['overlapThresh'] = 0.9
    paras['scoreThresh'] = 0.8
    paras['th'] = 0.15
    paras['region_patch_ratio'] = 0.05
    paras['imgSize'] = (440, 440)
    paras['nk'] = 1
    wordsFolder = '../../Data/Features/'
    wordsNum = 100
    patchSize = 16
    L = 100
    nk = 1  # k-means
    proposal_minSize = 100 * 100
    proposal_maxSize = 440 * 220
    paras['regionSizeRange'] = [proposal_minSize, proposal_maxSize]

    eraseMapPath = '../../Data/eraseMap.bmp'
    regionModelWeights = '../../Data/region_classification/output/vgg_cnn_m_1024_fast_rcnn_b500_iter_10000.caffemodel'
    regionModelPrototxt = '../../fast-rcnn/models/VGG_CNN_M_1024/test_kls.prototxt'
    if not os.path.exists(eraseMapPath):
        imSize = 440
        eraseMap = np.zeros((imSize, imSize))
        radius = imSize / 2
        centers = np.array([219.5, 219.5])
        for i in range(440):
            for j in range(440):
                if np.linalg.norm(np.array([i, j]) - centers) > 220 + 5:
                    eraseMap[i, j] = 1
        imsave(eraseMapPath, eraseMap)
    else:
        eraseMap = imread(eraseMapPath) / 255
    paras['eraseMap'] = eraseMap

    gpu_id = 0
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(regionModelPrototxt, regionModelWeights, caffe.TEST)
    paras['net'] = net

    # demo
    feaType = 'LBP'
    paras['types'] = ['arc', 'drapery', 'radial', 'hot_spot']  # 0: arc, 1: drapery, 2: radial, 3: hot-spot

    if feaType == 'LBP':
        paras['lbp_wordsFile_s1'] = wordsFolder + 'type4_LBPWords_s1_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'
        paras['lbp_wordsFile_s2'] = wordsFolder + 'type4_LBPWords_s2_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'
        paras['lbp_wordsFile_s3'] = wordsFolder + 'type4_LBPWords_s3_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'
        paras['lbp_wordsFile_s4'] = wordsFolder + 'type4_LBPWords_s4_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'
    if feaType == 'SIFT':
        paras['sift_wordsFile_s1'] = wordsFolder + 'type4_SIFTWords_s1_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'
        paras['sift_wordsFile_s2'] = wordsFolder + 'type4_SIFTWords_s2_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'
        paras['sift_wordsFile_s3'] = wordsFolder + 'type4_SIFTWords_s3_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'
        paras['sift_wordsFile_s4'] = wordsFolder + 'type4_SIFTWords_s4_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'
    if feaType == 'His':
        paras['his_wordsFile_s1'] = wordsFolder + 'type4_HisWords_s1_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'
        paras['his_wordsFile_s2'] = wordsFolder + 'type4_HisWords_s2_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'
        paras['his_wordsFile_s3'] = wordsFolder + 'type4_HisWords_s3_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'
        paras['his_wordsFile_s4'] = wordsFolder + 'type4_HisWords_s4_s' + str(patchSize) + '_b300_w' + str(wordsNum) + '.hdf5'

    imgFolder = '../../Data/images/'
    imgFiles = os.listdir(imgFolder)
    cls_time = 0
    seg_time = 0
    fig_id = 1
    for imName in imgFiles:
        imgFile = imgFolder + imName
        ### classification and KLS coarse localization
        cls_start = time.time()
        label, classHeatMap = ClsKLSCoarseLoc(imgFile, paras)  # 0: arc, 1: drapery, 2: radial, 3: hot-spot
        cls_end = time.time()

        cls_time += cls_end - cls_start


        ### KLS localization
        seg_start = time.time()

        kls = KLSLoc(label, classHeatMap, paras)

        seg_end = time.time()

        seg_time += seg_end-seg_start

        if True:  # show segmentation results
            # 0: arc, 1: drapery, 2: radial, 3: hot-spot
            class_names = ['background', 'arc', 'drapery', 'radial', 'hot-spot']
            im = paras['im']

            plt.figure(fig_id)

            ax1 = plt.subplot(131)
            ax1.imshow(im, cmap='gray')
            ax1.axis('off')
            plt.title('predict class: ' + class_names[label + 1])

            ax2 = plt.subplot(132)
            ax2.imshow(kls, cmap='gray')
            ax2.axis('off')
            plt.title('KLS location')

            img = paras['img']
            kls_color = np.zeros(img.shape, dtype='uint8')
            kls_color[:, :, 0][np.where(kls == 1)] = 255
            alpha = 0.2
            addImg = (kls_color * alpha + img * (1. - alpha)).astype(np.uint8)
            ax3 = plt.subplot(133)
            ax3.imshow(addImg)
            ax3.axis('off')
            plt.title('add image')

            fig_id += 1

    print "classification time: " + str(cls_time / len(imgFiles))
    print "segmentation time: " + str(seg_time / len(imgFiles))
    plt.show()

