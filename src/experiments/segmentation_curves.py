import matplotlib.pyplot as plt
from src.experiments.calCSAccuracy import calSegClsAccuracy
import numpy as np
import matplotlib

if __name__ == '__main__':
    feaTypes = ['LBP', 'SIFT', 'His']
    # feaTypes = ['LBP']
    wordsNums = [50, 100, 200, 500]  # , 800]
    patchSizes = [8, 16, 24, 32, 40, 48, 56, 64]
    # patchSizes = [16, 32, 48, 64]
    mks = [0]

    max_accuracy = 0

    resultSaveFolder = '../../Data/Results/segmentation/modelFS_segV2_FWP_mk0/'
    curves_dic = {}
    for patchSize in patchSizes:
        for wordsNum in wordsNums:
            for feaType in feaTypes:
                for mk in mks:
                    resultFile = resultSaveFolder + 'segmentation_' + feaType + '_w' + str(wordsNum) + '_s' + str(patchSize) + '_mk' + str(
                        mk) + '.txt'

                    key = feaType + '_w' + str(wordsNum)
                    if key not in curves_dic:
                        curves_dic[key] = []

                    cls_acc, seg_acc = calSegClsAccuracy(resultFile)
                    curves_dic[key].append(seg_acc.mean())
                    if seg_acc.mean() > max_accuracy:
                        max_accuracy = seg_acc.mean()
                    print feaType, wordsNum, patchSize, seg_acc.mean()
    print curves_dic
    print 'max accuracy: ' + str(max_accuracy)

    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=[8, 5])
    plt.xlabel('Patch size', fontsize=14)
    plt.ylabel('IoU', fontsize=14)
    keys = curves_dic.keys()
    keys.sort()

    lineTypes = {}
    lineTypes['LBP'] = 'o-'
    lineTypes['SIFT'] = '*--'
    lineTypes['His'] = 's:'

    for k in keys:
        v = curves_dic[k]
        # yticks = range(10, 110, 10)
        # ax.set_yticks(yticks)
        # ax.set_ylim([10, 110])
        # ax.set_xlim([58, 42])
        plt.plot(patchSizes, v, lineTypes[k.split('_')[0]], label=k)

    ax.set_ylim([0.35, 0.47])
    ax.set_xlim([8, 82])
    ax.set_xticks(np.linspace(8, 64, 8))
    ax.set_xticklabels(('8', '16', '24', '32', '40', '48', '56', '64'))
    ax.set_yticks(np.linspace(0.35, 0.47, 10))
    ax.set_facecolor('ghostwhite')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    plt.grid(color='white')
    # plt.grid(True)
    plt.legend( loc=7, borderaxespad=0., fontsize=11)
    plt.savefig('../../Data/Results/figs/modelFS_FWP_km0.pdf', bbox_inches='tight')
    plt.show()