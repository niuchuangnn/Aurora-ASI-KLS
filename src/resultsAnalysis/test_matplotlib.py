import matplotlib.pyplot as plt
from scipy.misc import imread

im = imread('/home/ljm/NiuChuang/cobe.jpg')
fig, ax1 = plt.subplots(figsize=(12, 12))
ax1.imshow(im)
ax1.add_patch(
    plt.Rectangle((0, 0), 100, 100, fill=True, facecolor='red', alpha=0.5)
)
plt.axis('off')
plt.tight_layout()
plt.show()
