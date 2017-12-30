from src.experiments.calCSAccuracy import calSegClsAccuracy

if __name__ == '__main__':
    resultFolder = '../../Data/Results/classification/'
    patchScales = ['S', 'L', 'F']

    for tr in patchScales:
        for te in patchScales:
            resultFile = resultFolder + 'result_classification_tr_' + tr + '_te_' + te + '.txt'
            acc = calSegClsAccuracy(resultFile)
            print tr, te, acc, acc.mean()