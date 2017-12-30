from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import src.util.paseLabeledFile as plf

def sizeVsRadius(r):
    max_size = 32.
    min_size = 16.
    resolution_center = 1.
    resolution_spherical = 18.
    image_radius = 220.

    ratio = (resolution_spherical - resolution_center) / image_radius
    size = max(max_size - (ratio * r), min_size)
    return size

def centerArr2sizeList(c, c_image=np.array([219.5, 219.5])):
    if len(c.shape) > 1:
        r = np.linalg.norm(c - c_image, axis=1)
        size_list = [sizeVsRadius(x) for x in r]
    else:
        r = np.linalg.norm(c - c_image)
        size_list = sizeVsRadius(r)
    return size_list

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


def generateGrid(imageSize, gridSize, sizeRange=(10, 30), diffResolution=False):
    """According to image size and grid size, this function generates a evenly grid
    that coded by upper-left coordinates and its with and height, finally returns these
    grids selected within the inscribed circle where aurora occurs.

    input: imagSize [H, W], grid size [h1 w1], sizeRange
    output: gridList"""
    [w_num, h_num] = np.floor(imageSize / gridSize)
    w_num = int(w_num)
    h_num = int(h_num)
    if diffResolution:
        x_map = np.tile(np.array(range(w_num)), [h_num, 1]) * gridSize[0] + 16
        y_map = np.tile(np.array(range(h_num)).reshape([h_num, 1]), [1, w_num]) * gridSize[1] + 16
    else:
        x_map = np.tile(np.array(range(w_num)), [h_num, 1]) * gridSize[0]
        y_map = np.tile(np.array(range(h_num)).reshape([h_num, 1]), [1, w_num]) * gridSize[1]
        w_map = np.random.randint(sizeRange[0], sizeRange[1] + 1, size=(h_num, w_num))
        # h_map = np.random.randint(sizeRange[0], sizeRange[1] + 1, size=(h_num, w_num))
        h_map = w_map

    gridList = []
    centers = np.array([(float(imageSize[0]) - 1) / 2, (float(imageSize[1]) - 1) / 2])  # index from 0
    radius = imageSize[0] / 2
    for i in range(h_num):
        for j in range(w_num):
            if diffResolution:
                c = np.array([[y_map[i, j], x_map[i, j]]])
                # r = np.linalg.norm(c - centers)
                # size = sizeVsRadius(r)
                size = centerArr2sizeList(c)[0]
                grid = (y_map[i, j]-(size/2), x_map[i, j]-(size/2), size, size)
            else:
                grid = (y_map[i, j], x_map[i, j], h_map[i, j], w_map[i, j])
            if isWithinCircle(grid, centers, radius):
                gridList.append(grid)
    return gridList


def generateGridPatchData(im, gridSize, sizeRange, imResize=None, gridList=None, imNorm=True, diffResolution=False):
    if isinstance(im, str):
        im = Image.open(im)
        if imResize:
            im = im.resize(imResize)

        im = np.array(im)

        if imNorm:
            im = np.array(im, dtype='f')/255

    if gridList is None:
        imageSize = np.array(im.shape)
        gridList = generateGrid(imageSize, gridSize, sizeRange, diffResolution=diffResolution)

    gridPatchData = []
    for grid in gridList:
        if im.ndim == 2:
            patch = im[grid[0]:(grid[0]+grid[2]), grid[1]:(grid[1]+grid[3])].copy()     # grid format: [y, x, h, w] (y: row, x: column)
        if im.ndim == 3:
            patch = im[grid[0]:(grid[0] + grid[2]), grid[1]:(grid[1] + grid[3]), :].copy()  # grid format: [y, x, h, w]
        gridPatchData.append(patch)
    return gridPatchData, gridList, im

if __name__ == '__main__':

    radius = range(1, 221)
    sizes = [sizeVsRadius(x) for x in radius]

    plt.plot(radius, sizes)
    plt.show()

    labelFile = '../../Data/balanceSampleFrom_one_in_minute.txt'
    imagesFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    gridSize = np.array([10, 10])
    sizeRange = (16, 16)

    [images, labels] = plf.parseNL(labelFile)

    # imgFile = imagesFolder + images[0] + imgType
    imName = 'N20031221G094901'
    saveFolder = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/Results/sample_grid/'
    imgFile = imagesFolder + imName + imgType

    # im = Image.open(imgFile)
    # im = np.array(im)

    # imageSize = np.array(im.shape)
    # gridList = generateGrid(imageSize, gridSize, sizeRange)
    gridPatchData, gridList, im = generateGridPatchData(imgFile, gridSize, sizeRange)

    # print gridList[0:5]
    # print len(gridList)
    plf.showGrid(im, gridList)
    plt.savefig(saveFolder+imName+'_grid.jpg')
    # plt.imsave('/home/ljm/NiuChuang/KLSA-auroral-images/Data/Results/sample_grid/N20040118G050641_grid.jpg')
    plt.show()
