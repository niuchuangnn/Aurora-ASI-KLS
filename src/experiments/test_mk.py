import numpy as np
import matplotlib.pyplot as plt
from src.experiments.calCSAccuracy import calSegClsAccuracy
import matplotlib

if __name__ == '__main__':
    resultsFolder = '../../Data/Results/segmentation/segV2_mk_LBP_s16_w500/'

    feaTypes = ['LBP']
    patchSizes = [16]
    wordsNums = [500]
    mks = range(0, 7000, 400)
    auroraTypes = ['Arc', 'Drapery', 'Radial', 'Hot-spot']
    classNum = 4
    result_arr = np.zeros((len(mks), classNum))

    for feaType in feaTypes:
        for patchSize in patchSizes:
            for wordsNum in wordsNums:
                for i in range(len(mks)):
                    mk = mks[i]
                    resultFile = resultsFolder + 'segmentation_' + feaType + '_w' + str(wordsNum) + '_s' + str(patchSize) + '_mk' + str(mk) + '.txt'
                    cls_acc, seg_acc = calSegClsAccuracy(resultFile)
                    result_arr[i] = seg_acc
                    # print cls_acc, seg_acc
    # print result_arr
    #
    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=[8, 5])
    plt.xlabel(r'$L$', fontsize=14)
    plt.ylabel('IoU', fontsize=14)

    lineTypes = []
    lineTypes.append('o-')
    lineTypes.append('*-')
    lineTypes.append('s-')
    lineTypes.append('v-')

    for i in range(classNum):
        plt.plot(mks, result_arr[:, i], lineTypes[i], label=auroraTypes[i])

    ax.set_ylim([0.35, 0.46])
    ax.set_xlim([0, 6800])
    ax.set_xticks(np.linspace(0, 6800, 5))
    ax.set_yticks(np.linspace(0, 0.57, 8))
    ax.set_facecolor('ghostwhite')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    fig.set_edgecolor('none')
    plt.grid(color='white')
    # plt.tight_layout(pad=0)
    plt.legend(bbox_to_anchor=(0.28, 0.33), borderaxespad=0., loc=1, fontsize=14)
    plt.savefig('../../Data/Results/figs/LBP_s16_w' + str(wordsNums[0]) + '.pdf', egdecolor='white')
    plt.show()