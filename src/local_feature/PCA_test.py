import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

colors = ['blue', 'green']
x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([0, 0, 0, 1, 1, 1])
_, axes = plt.subplots(1, 3, figsize=(21, 7))
axes[0].scatter(x[:, 0], x[:, 1], c=y.astype(np.float), s=40)
axes[0].set_xticks(np.linspace(-4, 4, 9))
axes[0].set_yticks(np.linspace(-4, 4, 9))
pca = PCA(n_components=2)
pca.fit(x)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
u = pca.components_
print u
x_rot = -x.dot(u.T)
print x_rot

axes[1].scatter(x_rot[:, 0], x_rot[:, 1], c=y.astype(np.float), s=40)
axes[1].set_xticks(np.linspace(-4, 4, 9))
axes[1].set_yticks(np.linspace(-4, 4, 9))

axes[2].scatter(x_rot[:, 0], np.zeros(x_rot[:, 1].shape), c=y.astype(np.float), s=40)
axes[2].set_xticks(np.linspace(-4, 4, 9))
axes[2].set_yticks(np.linspace(-4, 4, 9))
plt.show()