from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    im = Image.open('/home/ljm/NiuChuang/cobe.jpg')
    print im.format, im.size, im.mode
    im = im.resize((256, 256))
    im = np.array(im)
    plt.imshow(im)
    plt.show()