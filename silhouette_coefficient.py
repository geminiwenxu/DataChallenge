from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score



def bench_k_means(method, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), method).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))


def quality(name, X, labels):
    if name == "KMeans":
        method = KMeans(n_clusters=3, random_state=0).fit(X)
        #bench_k_means(method, "KMeans", X, labels)
        #contingency_matrix = metrics.cluster.contingency_matrix(X, method)
          # return purity
    
    
    #return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix) 









if __name__ == "__main__":
    from sklearn.cluster import KMeans
    import pandas as pd
    from sklearn.cluster import MeanShift

    df = pd.read_csv('/Users/smile/Desktop/Task3_data/age_sub_20_40.csv', sep=',').fillna(0)
    X = df.to_numpy()
    labels = df['SepsisLabel']

    method = MeanShift(bandwidth=200).fit(X)
    print(method.labels_)
    print(labels)
    print(purity_score(labels, method.labels_))
    #quality("KMeans", X, labels)

    