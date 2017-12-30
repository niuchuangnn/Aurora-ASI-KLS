import numpy as np

def normalizeVecs(vecs):
    len_vecs = np.sqrt(np.sum(vecs ** 2, axis=1))
    vecs = vecs / len_vecs.reshape((len_vecs.size, 1))
    return vecs

if __name__ == '__main__':
    v = np.array([[1,1,1],
                  [2,2,2],
                  [3,3,3]])
    print normalizeVecs(v)