import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from scipy.cluster.hierarchy import dendrogram, linkage

if __name__ == '__main__':
    X = read_csv('votes.csv', sep=',').to_numpy()
    X = np.nan_to_num(X, nan=0)
    plt.figure(figsize=(20, 20))
    plt.grid(True)
    plt.title('Hierarchical Clustering Dendrogram')
    Z = linkage(X)
    dendrogram(Z, truncate_mode='level', p=10)
    plt.show()
