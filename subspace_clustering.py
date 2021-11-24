import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score
import pandas as pd


def prepare_data(path):
    df = pd.read_csv(path, sep=',').fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    X = df.to_numpy()
    plt.matshow(X, cmap=plt.cm.Blues, interpolation="none")
    plt.title("Original dataset")
    # plt.show()
    return X


def subspace_clustering(n_cluster, X, name):
    if name == 'SpectralBiclustering':
        model = SpectralBiclustering(n_clusters=n_cluster, random_state=0).fit(X)
    elif name == 'SpectralCoclustering':
        model = SpectralCoclustering(n_clusters=n_cluster, random_state=0).fit(X)
    fit_data = X[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]
    plt.matshow(fit_data, cmap=plt.cm.Blues)
    plt.title("After biclustering; rearranged to show biclusters")

    plt.matshow(np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1),
                cmap=plt.cm.Blues)
    plt.title("Checkerboard structure of rearranged data")
    # plt.show()


if __name__ == '__main__':
    X = prepare_data("/Users/wenxu/PycharmProjects/DataChallenge/data/task_3/dr_age_sub_20_40.csv")
    print(X)
    subspace_clustering(3, X, 'SpectralCoclustering')
