import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage.transform import rotate
import numpy as np

def show_kls(klsdb):
    img_num = len(klsdb)
    for i in xrange(img_num):
        imi = klsdb[i]
        imFile = imi['image']
        im = imread(imFile)
        angle = imi['angle']
        im = rotate(im, angle, preserve_range=True)
        im = im.astype(np.uint8)

        bboxes = imi['bbox']
        labels = imi['gt_classes']


        for j in xrange(bboxes.shape[0]):
            plt.imshow(im, cmap='gray')
            roi = bboxes[j, :]
            plt.gca().add_patch(
                plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                              roi[3] - roi[1], fill=False,
                              edgecolor='r', linewidth=3)
            )
            plt.title(str(labels[j]))
            plt.show()