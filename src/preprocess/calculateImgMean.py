import numpy as np
from src.util.paseLabeledFile import parseNL
from scipy.misc import imread

if __name__ == '__main__':
    labelFile = '../../Data/Alllabel2003_38044.txt'
    imgsFolder = '../../Data/all38044JPG/'
    imgType = '.jpg'
    [names, labels] = parseNL(labelFile)

    img_num = len(names)
    mean = 0
    for i in range(img_num):
        name = names[i]

        imgFile =imgsFolder + name + imgType
        im = imread(imgFile)
        mean += im.mean()
    mean = mean/img_num

    print mean