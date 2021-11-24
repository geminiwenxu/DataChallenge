import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import partition
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score
import pandas as pd
from sklearn.datasets import make_checkerboard


def prepare_data(path):
    df = pd.read_csv(path, sep=',').fillna(0)
    X = df.to_numpy()
    plt.matshow(X, cmap=plt.cm.Blues, interpolation="none")
    plt.title("Original dataset")
    plt.show()
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
    plt.show()
    rng = np.random.RandomState(0)
    row_idx = rng.permutation(X.shape[0])
    col_idx = rng.permutation(X.shape[1])

    data, rows, columns = make_checkerboard(
        shape=(X.shape[0], X.shape[1]), n_clusters=n_cluster, random_state=0

    )
    print(rows[0])
    score = consensus_score(model.biclusters_, (rows[:, row_idx], columns[:, col_idx]))
    print("consensus score: {:.1f}".format(score))


if __name__ == '__main__':
    X = prepare_data("//training_test")
    subspace_clustering(3, X, 'SpectralCoclustering')
