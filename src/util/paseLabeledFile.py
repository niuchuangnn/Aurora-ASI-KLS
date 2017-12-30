import datetime
import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np

def parseNL(path):
    f = open(path, 'r')
    names = []
    labels = []
    lines = f.readlines()
    for line in lines:
        if len(line.split()) == 1:
            name = line.split()
            names.append(name[0])
        if len(line.split()) == 2:
            [name, label] = line.split()
            names.append(name)
            labels.append(label)
    if len(labels) == 0:
        return names
    else:
        return names, labels

def timeDiff(name1, name2):
    # formate: N20031221G030001
    date1 = datetime.datetime(int(name1[1:5]), int(name1[5:7]), int(name1[7:9]), int(name1[10:12]), int(name1[12:14]), int(name1[14:16]))
    date2 = datetime.datetime(int(name2[1:5]), int(name2[5:7]), int(name2[7:9]), int(name2[10:12]), int(name2[12:14]), int(name2[14:16]))
    return (date2-date1).seconds

def sampleImages(names, mode='Uniform', timediff = 60, sampleNum = 500):
    if mode == 'Uniform':
        lastImg = names[0]
        sampledImgs = []
        ids = []
        # sampledLabels = []
        sampledImgs.append(lastImg)
        ids.append(0)
        # sampledLabels.append(labels[0])
        id = 0
        for name in names:
            if timeDiff(sampledImgs[-1], name) >= timediff:
                sampledImgs.append(name)
                ids.append(id)
                # sampledLabels.append(labels[id])
            id = id + 1
        return ids, sampledImgs
    if mode == 'random':
        import random
        # sampledImgs = []
        # ids = []
        # idx = range(len(names))
        random.shuffle(names)
        # for i in range(sampleNum):
        #     sampledImgs.append(names[idx[i]])
            # ids.append(idx[i])
        return names[:sampleNum]

def writeArrangeImgsToFile(arrangeImgs, filePath, labelAdjust=None, addType=None):
    f = open(filePath, 'w')
    for label, images in arrangeImgs.iteritems():
        if labelAdjust is not None:
            label = str(int(label) + labelAdjust)
        for name in images:
            if addType is not None:
                name += addType
            f.write(name + ' ' + label + '\n')
    f.close()
    return 0

def arrangeToClasses(names, labels, classNum=4, classLabel=[['1'], ['2'], ['3'], ['4']]):
    arrangeImgs = {}
    rawTypes = {}
    for i in range(classNum):
        arrangeImgs[str(i+1)] = []
        rawTypes[str(i+1)] = []

    for i in range(len(names)):
        for j in range(classNum):
            if labels[i] in classLabel[j]:
                arrangeImgs[str(j+1)].append(names[i])
                rawTypes[str(j+1)].append(labels[i])
    if classNum == 4:
        return arrangeImgs
    if classNum < 4:
        return arrangeImgs, rawTypes

def arrangeToDays(names, labels):
    name_days = {}
    label_days = {}
    days = []
    names_num = len(names)
    for i in xrange(names_num):
        name = names[i]
        label = labels[i]
        day = name[1:9]

        if day not in name_days:
            name_days[day] = []
            label_days[day] = []
            days.append(day)
        name_days[day].append(name)
        label_days[day].append(label)
    return name_days, label_days, days

def splitConfigFile(sourceFile, savePathes=['../../Data/train_day16.txt', '../../Data/test_day3.txt'],
                    splitGroups=[range(15), range(15, 19)], isBalabceSamples=[True, False], labelAdjust=None, addType=None):
    [names, labels] = parseNL(sourceFile)
    names_days, labels_days, days = arrangeToDays(names, labels)

    groups_num = len(splitGroups)
    for i in xrange(groups_num):
        group_i = splitGroups[i]
        savePath_i = savePathes[i]
        isBalance_i = isBalabceSamples[i]
        group_i_dayNum = len(group_i)
        names_i = []
        labels_i = []
        for j in xrange(group_i_dayNum):
            names_i += names_days[days[group_i[j]]]
            labels_i += labels_days[days[group_i[j]]]
        arrangeImgs_i = arrangeToClasses(names_i, labels_i, classNum=4, classLabel=[['1'], ['2'], ['3'], ['4']])
        if isBalance_i:
            max_balance_num = min([len(x) for x in arrangeImgs_i.values()])
            arrangeImgs_i = balanceSample(arrangeImgs_i, max_balance_num)
        writeArrangeImgsToFile(arrangeImgs_i, savePath_i, labelAdjust, addType)
    return 0

def balanceSample(arrangedImgs, sampleNum):
    for c in arrangedImgs:
        arrangedImgs[c] = sampleImages(arrangedImgs[c], mode='random', sampleNum=sampleNum)
    return arrangedImgs

def compareLabeledFile(file_std, file_compare, labelAdjust=None, addType=None):
    [names_std, labels_std] = parseNL(file_std)
    [names_compare, labels_compare] = parseNL(file_compare)
    flag = True
    for i in range(len(names_compare)):
        name_c = names_compare[i]
        if addType is not None:
            name_c = name_c[:-len(addType)]
        std_idx = names_std.index(name_c)
        label_c = labels_compare[i]
        if labelAdjust is not None:
            label_c = str(int(label_c)+labelAdjust)
        label_std = labels_std[std_idx]
        if label_std != label_c:
            flag = False
            break
    return  flag

def findTypes(sourceFile, names):
    [sourceNames, sourceTypes] = parseNL(sourceFile)
    types = []
    for n in names:
        idx = sourceNames.index(n)
        types.append(sourceTypes[idx])
    return types

def splitToClasses(sourceFile, names):
    types = findTypes(sourceFile, names)
    cs = set(types)
    # typesNum = len(cs)
    splitImgs = {}
    for i in cs:
        splitImgs[i] = []
    for j in range(len(names)):
        splitImgs[types[j]].append(names[j])
    return splitImgs

def showGrid(im, gridList):
    fig, ax = plt.subplots(figsize=(12, 12))
    if len(im.shape) == 2:
        ax.imshow(im, aspect='equal', cmap='gray')
    else:
        ax.imshow(im, aspect='equal')
    for grid in gridList:
        ax.add_patch(
            plt.Rectangle((grid[1], grid[0]),
                          grid[3], grid[2],
                          fill=False, edgecolor='yellow',
                          linewidth=1)
        )
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def showProposals(im, proposals):
    # box format: [x1, x2, y1, y2]
    fig, ax = plt.subplots(figsize=(12, 12))
    if len(im.shape) == 2:
        ax.imshow(im, aspect='equal', cmap='gray')
    else:
        ax.imshow(im, aspect='equal')
    for i in xrange(proposals.shape[0]):
        box_i = proposals[i, :]
        ax.add_patch(
            plt.Rectangle((box_i[0], box_i[1]),
                          box_i[2] - box_i[0],
                          box_i[3] - box_i[1],
                          fill=False, edgecolor='yellow',
                          linewidth=0.35)
        )
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def calculateImgsMean(imgsFile, dataFolder, imgType=None):
    [names, labels] = parseNL(imgsFile)
    mean = 0.
    for name in names:
        if imgType is not None:
            imgFile = dataFolder + name + imgType
        else:
            imgFile = dataFolder + name
        im = imread(imgFile)
        mean += im.mean()
    mean = mean / len(names)
    print imgsFile + ' mean: ' + str(mean)
    return mean
def adjustLables(labelFile, adjust=-1):
    [names, labels] = parseNL(labelFile)
    labels_num = [int(x) for x in labels]
    labels_num = list(np.array(labels_num) + adjust)
    labels = [str(x) for x in labels_num]
    f = open(labelFile, 'w')
    for i in xrange(len(labels)):
        name = names[i]
        label = labels[i]
        f.write(name + ' ' + label + '\n')
    f.close()
    return 0

def selectTypeImages(labelFile, select_types, savePath=None, withExt=None):
    [names, labels] = parseNL(labelFile)
    imgs_dic = arrangeToClasses(names, labels)
    imgs_select = {}
    for t in select_types:
        imgs_select[t] = imgs_dic[t]
    if savePath is not None:
        f = open(savePath, 'w')
        for k, v in imgs_select.iteritems():
            for img in v:
                if withExt is not None:
                    f.write(img + withExt + ' ' + k + '\n')
                else:
                    f.write(img + ' ' + k + '\n')
        f.close()
    return imgs_select

if __name__ == '__main__':
    allLabels = '../../Data/Alllabel2003_38044.txt'
    savePathes = ['../../Data/train_day16_all.txt', '../../Data/test_day3_all.txt']
    imgType = '.jpg'
    # days = ['20031221', '20031222', '20031223', '20031224', '20031225', '20031226', '20031227', '20031228', '20031229',
    #  '20031230', '20031231', '20040101', '20040102', '20040103', '20040112', '20040114', '20040116', '20040117',
    #  '20040118']
    [names, labels] = parseNL(allLabels)

    names_days, labels_days, days = arrangeToDays(names, labels)
    splitConfigFile(allLabels, savePathes=savePathes, labelAdjust=-1, addType=imgType, isBalabceSamples=[True, False])
    # calculateImgsMean(savePathes[0], dataFolder='../../Data/all38044JPG/')
    print compareLabeledFile(allLabels, savePathes[0], labelAdjust=1, addType=imgType)
    print compareLabeledFile(allLabels, savePathes[1], labelAdjust=1, addType=imgType)
    print days
    # labelFile1 = '../../Data/train_16_3_7_2663.txt'
    # adjustLables(labelFile1)
    # select = '../../Data/all_arc.txt'
    # selectTypeImages(allLabels, ['1'], select, withExt='.jpg')
    pass