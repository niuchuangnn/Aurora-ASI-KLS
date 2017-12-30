import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import math
from scipy import signal
import os
import src.util.paseLabeledFile as plf

def calculateImgHist(imgFile, imResize=None, imSize=440):
    im = imread(imgFile)
    if imResize is not None:
        im = imresize(im, [imResize, imResize])
        imSize = imResize
    outCircleSize = int(round(((imSize * imSize) - (math.pi * ((imSize/2 + 1) ** 2)))))

    bins = range(257)
    arr = im.flatten()
    hist, _ = np.histogram(arr, bins=bins)
    hist[0] -= outCircleSize
    return hist

def adaptiveLinear(x):
    maxTh = 80
    a = 0.005
    b = 25
    y = min(b + a * x, maxTh)
    return y

def calAdapList(xl):
    yl = [adaptiveLinear(x) for x in xl]
    return yl

def calculateThreshold(imgFile, imResize=None):
    hist = calculateImgHist(imgFile, imResize=imResize)
    thresh = adaptiveLinear(hist[180:].sum())
    return thresh

if __name__ == '__main__':
    imgFile = '/home/amax/NiuChuang/KLSA-auroral-images/Data/all38044JPG/N20040117G130515.jpg'
    # thresholdLabelFile = '../../Data/threshold_annotations.txt'
    # imagesFolder_thresh = '../../Data/images_for_threshold/'
    # imgType = '.jpg'
    # [names, labels] = plf.parseNL(thresholdLabelFile)
    # bright_nums = np.zeros((len(names), ))
    # thresholds = np.zeros((len(names), ))
    # for i in xrange(len(names)):
    #     imgFile = imagesFolder_thresh + names[i] + imgType
    #     hist = calculateImgHist(imgFile)
    #     bright_nums[i] = hist[150:].sum()
    #     thresholds[i] = int(labels[i])
    # plt.figure(1)
    # xs = range(60000)
    # ys = calAdapList(xs)
    # plt.plot(xs, ys)
    # plt.scatter(bright_nums, thresholds, 50, color='blue')
    # plt.show()
    outCircleSize = int(round(((440*440) - (math.pi * 221 * 221))))
    im = imread(imgFile)
    bins = range(0, 256)
    arr = im.flatten()
    hist, _ = np.histogram(arr, bins=bins)
    # plt.figure(1)
    # plt.plot(range(255), list(hist))
    hist[0] -= outCircleSize
    plt.figure(0)
    plt.hist(arr, bins=255, normed=1, edgecolor='None', facecolor='red')
    plt.figure(1)
    plt.imshow(im, cmap='gray')
    plt.show()
    # imagesFolder = '../../Data/images_for_threshold/'
    # files = os.listdir(imagesFolder)
    # for fn in files:
    #     imgFile = imagesFolder + fn
    #     print fn
    #
    #     fig, axes = plt.subplots(1, 3, figsize=[21, 7])
    #
    #     hist = calculateImgHist(imgFile)
    #     # print len(hist), hist[len(hist)-1]
    #     # plt.figure(1)
    #     peakind_hist = signal.find_peaks_cwt(hist, np.arange(1, 129), min_snr=5)
    #     axes[0].plot(range(256), list(hist))
    #     axes[0].scatter(np.array(range(256))[peakind_hist], hist[peakind_hist], 50, color='blue')
    #     print peakind_hist
    #
    #     # xs = np.arange(0, np.pi, 0.05)
    #     # data = np.sin(xs)
    #     # peakind = signal.find_peaks_cwt(data, np.arange(1, 1000))
    #     # plt.figure(2)
    #     # plt.plot(xs, data)
    #     # plt.scatter(xs[peakind], data[peakind], 50, color='blue')
    #     # print peakind, xs[peakind], data[peakind]
    #
    #     # plt.figure(3)
    #     peakind_hist = signal.find_peaks_cwt(-hist[15:], np.arange(1, 129), min_snr=5)
    #     axes[1].plot(range(256-15), list(-hist[15:]))
    #     axes[1].scatter(np.array(range(256-15))[peakind_hist], -hist[15:][peakind_hist], 50, color='blue')
    #     print np.array(peakind_hist) + 15
    #
    #     # plt.figure(4)
    #     im = imread(imgFile)
    #     axes[2].imshow(im, cmap='gray')
    #     axes[2].axis('off')
    #
    #     plt.show()