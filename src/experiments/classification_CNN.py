import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../fast-rcnn/caffe-fast-rcnn')
import caffe
import os
import skimage.transform as trans
from skimage.io import imread
import random
import time

def CNN_classification(im_trans, net):
    net.blobs['data'].data[...] = im_trans
    output = net.forward()
    output_prob = output['prob'][0]
    return output_prob.argmax()

if __name__ == '__main__':
    imgFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/alllbp04-09Pri_70K_selected/1/N20080109G133700.jpg'
    imSize = 224
    caffe.set_device(1)
    caffe. set_mode_gpu()
    # model_def = '../../Data/region_classification/output/deploy.prototxt'
    # model_weights = '../../Data/region_classification/output/aurora_whole_iter_10000.caffemodel'
    # model_def = '../../Data/region_classification/output/deploy_cam_vgg16.prototxt'
    # model_weights = '../../Data/region_classification/output/vgg16_cam_iter_10000.caffemodel'
    model_def = '../../Data/region_classification/output/deploy_vgg16.prototxt'
    model_weights = '../../Data/region_classification/output/vgg_16_iter_40000.caffemodel'
    # meanFileOrValue = '../../Data/region_classification/output/imgMean38044.npy'
    meanFileOrValue = 35.37
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_raw_scale('data', 255)
    net.blobs['data'].reshape(1, 1, imSize, imSize)
    # im = imread(imgFile)
    # im = trans.resize(im, (256, 256))
    # mu = np.load(meanFile)[0]
    # im = im - mu
    # # print im.shape, im.max(), im.min()
    # net.blobs['data'].reshape(10, 1, 256, 256)
    # image = caffe.io.load_image(imgFile)
    # print image.shape, image.max(), image.min()
    # print mu.shape, mu.max(), mu.min()
    # transformed_im = transformer.preprocess('data', im)
    # print transformed_im.shape, transformed_im.max(), transformed_im.min()

    # label = CNN_classification(transformed_im, net)
    # print label

    resultsSaveFolder = '../../Data/Results/classification/'
    result_cls = 'result_classification_vgg16.txt'
    classNum = 4
    confusionArray_c = np.zeros((classNum, classNum))
    IoU_accuracy = np.zeros((classNum,))
    labelDataFolder = '../../Data/alllbp04-09Pri_70K_selected/'
    # imgFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/N20031222G074652.bmp'

    f_cls = open(resultsSaveFolder + result_cls, 'w')
    test_num_per_class = 500
    for c in xrange(0, classNum):
        labelImgFolder_c = labelDataFolder + str(c + 1)
        imgFiles = os.listdir(labelImgFolder_c)
        random.seed(10)
        random.shuffle(imgFiles)

        for im_idx in range(test_num_per_class):
            start = time.time()
            imgName = imgFiles[im_idx]
            imgFile = labelImgFolder_c + '/' + imgName
            imName = imgName[:-4]
            print imName
            im = imread(imgFile)
            im = trans.resize(im, (imSize, imSize))
            if isinstance(meanFileOrValue, str):
                mu = np.load(meanFileOrValue)[0]
                mu = trans.resize(mu, (imSize, imSize))
                im = im - mu
            # print im.shape, im.max(), im.min()

            transformed_im = transformer.preprocess('data', im)
            if not isinstance(meanFileOrValue, str):
                transformed_im -= 35.37
            label = CNN_classification(transformed_im, net)
            end = time.time()

            print end-start

            confusionArray_c[c, label] += 1
            f_cls.write(imName + ' ' + str(c) + ' ' + str(label) + '\n')
    print confusionArray_c
    f_cls.close()
    accuracy = confusionArray_c / np.sum(confusionArray_c, axis=1).reshape(classNum, 1)
    print accuracy