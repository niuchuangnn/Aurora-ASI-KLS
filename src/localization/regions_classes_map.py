import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../../selective_search_py')
sys.path.insert(0, '../../fast-rcnn/lib')
sys.path.insert(0, '../../fast-rcnn/caffe-fast-rcnn/python')
import caffe
from fast_rcnn.test_kls import im_detect, vis_detections
import selective_search as ss
import skimage.io
import features
import os
from scipy.misc import imread, imsave
from src.preprocess.proposal_ss import filterOverlap
import color_space
# import selective_search
import src.util.paseLabeledFile as plf
import segment
import src.preprocess.esg as esg

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

def visRegionClassHeatMap(regionClassHeatMap, class_names):
    for i in xrange(len(class_names)):
        heatMap_i = regionClassHeatMap[:, :, i] * 255
        plt.figure(i)
        plt.imshow(heatMap_i, cmap='gray')
        plt.title(class_names[i])
    plt.show()

def region_class_heatMap(paras):
    img = paras['img']
    im = paras['im']
    color_space = paras['color_space']
    ks = paras['ks']
    feature_masks = paras['feature_masks']
    eraseMap = paras['eraseMap']
    overlapThresh = paras['overlapThresh']
    regionSizeRange = paras['regionSizeRange']
    net = paras['net']
    scoreThresh = paras['scoreThresh']
    is_showProposals = paras['is_showProposals']
    region_set = ss.selective_search(img, color_spaces=color_space, ks=ks,
                                     feature_masks=feature_masks, eraseMap=eraseMap)
    boxes = regionSetToBoxes(region_set, overlapThresh, sizeRange=regionSizeRange, isVisualize=False)
    if is_showProposals:
        plf.showProposals(im, boxes)
        plt.savefig('../../Data/Results/proposals/'+paras['imgFile'][-20:-4]+'_proposals.jpg')
        plt.show()
    scores, boxes = im_detect(net, im, boxes)
    regionClassMap = generateRegionClassHeatMap(scores, boxes, scoreThresh)
    return regionClassMap

if __name__ == '__main__':
    imgFile = '../../Data/labeled2003_38044/N20031222G072122.bmp'
    color_space = ['rgb']
    ks = [25, 50, 100, 150, 200]
    feature_masks = [1, 1, 1, 1]
    overlapThresh = 0.7
    scoreThresh = 0.9
    gpu_id = 1
    eraseMapPath = '../../Data/eraseMap.bmp'
    regionModelWeights = '../../Data/region_classification/output/vgg_cnn_m_1024_fast_rcnn_iter_10000.caffemodel'
    regionModelPrototxt = '../../fast-rcnn/models/VGG_CNN_M_1024/test_kls.prototxt'
    proposal_minSize = 100 * 100
    proposal_maxSize = 440 * 220
    regionSizeRange = [proposal_minSize, proposal_maxSize]
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
    im = skimage.io.imread(imgFile)
    if len(im.shape) == 2:
        img = skimage.color.gray2rgb(im)
    region_set = ss.selective_search(img, color_spaces=color_space, ks=ks,
                                     feature_masks=feature_masks, eraseMap=eraseMap)
    boxes = regionSetToBoxes(region_set, overlapThresh, sizeRange=regionSizeRange, isVisualize=False)

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(regionModelPrototxt, regionModelWeights, caffe.TEST)

    scores, boxes = im_detect(net, im, boxes)
    regionClassMap = generateRegionClassHeatMap(scores, boxes, scoreThresh)

    class_names = ['background', 'arc', 'drapery', 'radial', 'hot-spot']
    visRegionClassHeatMap(regionClassMap, class_names)
    vis_detections(im, scores, boxes, class_names)
    pass
