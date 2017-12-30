import numpy as np
import matplotlib.pyplot as plt

def parseResultFile(resultFile):
    f = open(resultFile, 'r')
    lines = f.readlines()
    imageNames = []
    trueLabels = []
    predictLabels = []
    IoUs = []
    for l in lines:
        result = l.split()
        imageNames.append(result[0])
        trueLabels.append(int(result[1]))
        predictLabels.append(int(result[2]))
        if len(result) >= 4:
            IoUs.append(float(result[3]))
    return imageNames, trueLabels, predictLabels, IoUs

def calSegClsAccuracy(resultFile, class_num=4):
    imageNames, trueLabels, predictLabels, IoUs = parseResultFile(resultFile)
    trueLabels_arr = np.array(trueLabels)
    predictLabels_arr = np.array(predictLabels)

    confusionMtrx = np.zeros((class_num, class_num))

    if len(IoUs) == 0:
        confusion_images_file = '../../Data/Results/classification/confusion_images.txt'
        f_confusion = open(confusion_images_file, 'w')
        for i in xrange(len(imageNames)):
            confusionMtrx[trueLabels_arr[i], predictLabels_arr[i]] += 1
            if predictLabels_arr[i] != trueLabels_arr[i]:
                f_confusion.write(imageNames[i] + ' ' + str(trueLabels_arr[i]) + ' ' + str(predictLabels_arr[i]) + '\n')
        rightNums = np.array([confusionMtrx[x, x] for x in range(class_num)], dtype='f')
        classification_arr = rightNums / np.sum(confusionMtrx, axis=1)
        f_confusion.close()
        # print confusionMtrx
        return classification_arr
    else:
        segmentation_acc_sum = np.zeros((class_num,))
        IoUs_arr = np.array(IoUs)

        for i in xrange(len(imageNames)):
            confusionMtrx[trueLabels_arr[i], predictLabels_arr[i]] += 1
            if trueLabels_arr[i] == predictLabels_arr[i]:
                segmentation_acc_sum[trueLabels_arr[i]] += IoUs_arr[i]

        rightNums = np.array([confusionMtrx[x, x] for x in range(class_num)], dtype='f')
        classification_arr = rightNums / np.sum(confusionMtrx, axis=1)
        segmentation_acc_true = segmentation_acc_sum / rightNums
        return classification_arr, segmentation_acc_true

if __name__ == '__main__':
    # resultFile_seg = '../../Data/Results/segmentation/segV2_LBP_s16_w100_mk0_noDetection/segmentation_LBP_w100_s16_mk0_noDetection.txt'
    # resultFile_seg = '../../Data/Results/segmentation/fcn/fcn_32s.txt'
    resultFile_seg = '../../Data/Results/segmentation/CAM/vgg16_th0.5_nodetection.txt'
    # resultFile_seg = '../../Data/Results/segmentation/RD/rd.txt'
    # resultFile_seg = '../../Data/Results/segmentation/segV2_LBP_s16_w100_mk0_nk/segmentation_LBP_w100_s16_mk0nk_9.txt'
    c_arr, s_arr = calSegClsAccuracy(resultFile_seg)
    print c_arr
    print s_arr
    print s_arr.mean()
    resultFile_cls = '../../Data/Results/classification/result_classification_CAM.txt'
    cls_arr = calSegClsAccuracy(resultFile_cls)
    print cls_arr
    print cls_arr.mean()