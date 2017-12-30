from skimage import feature
import src.util.paseLabeledFile as plf
import src.util.normalizeVecs as nv
import src.preprocess.esg as esg
import numpy as np
import h5py

def intensityFeature(imgFile=None, gridSize=None, sizeRange=None , gridPatchData=None, imResize=None, gridList=None, diffResolution=True):
    if gridPatchData is None:
        if imResize:
            gridPatchData, positions, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange, imResize=imResize, gridList=gridList)
        else:
            gridPatchData, positions, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange, gridList=gridList, diffResolution=diffResolution)

    if not diffResolution:
        gridPatchData_v = np.array(gridPatchData)
        patch_num = len(gridPatchData)
        mean_v = np.mean(np.mean(gridPatchData_v, axis=1), axis=1).reshape((patch_num, 1))

        # i = 16
        # patch_i = gridPatchData[i]
        # mean_i = patch_i.mean()
        shape = gridPatchData_v.shape
        gridPatchData_reshape = gridPatchData_v.reshape((shape[0], shape[1]*shape[2]))
        min_v = np.min(gridPatchData_reshape, axis=1).reshape((patch_num, 1))
        max_v = np.max(gridPatchData_reshape, axis=1).reshape((patch_num, 1))
        # min_i = patch_i.min()
        # max_i = patch_i.max()
        intensityFeas = np.hstack((min_v, mean_v, max_v))
    else:
        i = 0
        patch_i = gridPatchData[i]
        mean_i = patch_i.mean()
        # print mean_i
        feas_list = [[np.min(x), np.mean(x), np.max(x)] for x in gridPatchData]
        intensityFeas = np.array(feas_list)
        mean_i = intensityFeas[i, 1]
        # print mean_i
    # print min_v.shape
    # print gridPatchData_v.shape
    # print mean_i, mean_v[i], min_i, min_v[i], max_i, max_v[i]
    # print intensityFeas.min(), intensityFeas.max()
    return intensityFeas

if __name__ == '__main__':
    gridSize = np.array([10, 10])
    sizeRange = (16, 16)
    imgFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/N20031221G070851.bmp'
    feas = intensityFeature(imgFile, gridSize, sizeRange)
    print feas.shape
    print feas.min(), feas.max()
    # print feas