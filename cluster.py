import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation


def cluster(cluster, X, n=None):
    if cluster == 'KMeans':  # Wenxu
        kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
        labels = kmeans.labels_
        ls_cluster = np.unique(labels)

    elif cluster == 'MeanShift':  # Xiaoyan
        clustering = MeanShift(bandwidth=200).fit(X)
        labels = clustering.labels_
        ls_cluster = np.unique(labels)

    elif cluster == 'AffinityPropagation':
        clustering = AffinityPropagation(random_state=5).fit(X)
        labels = clustering.labels_
        ls_cluster = np.unique(labels)
        print(labels)

    # elif cluster == 'OPTICS':
    #     clustering = OPTICS(min_samples=2).fit(X)
    #     labels = clustering.labels_
    # elif cluster == 'SpectralClustering':
    #     clustering = SpectralClustering(n_clusters=n, assign_labels='discretize', random_state=0).fit(X)
    #     labels = clustering.labels_
    return labels, ls_cluster


def plot(cluster, df, ls_cluster):
    if cluster == "KMeans" or "AffinityPropagation":
        for i in range(len(ls_cluster)):
            filtered_label = df[df.label == i]
            plt.scatter(filtered_label['x'], filtered_label['y'])
    elif cluster == "MeanShift":
        for i in range(len(ls_cluster)):
            filtered_label = df[df.label == i]
            plt.scatter(filtered_label['HR'], filtered_label['Age'])

    plt.show()





if __name__ == '__main__':
    # read data and store it in a dataframe
    df = pd.read_csv('/training_test',
                     sep=',').fillna(0)
    # fill nan with 0 and transfer the dataframe into numpy array
    X = df.to_numpy()
    # apply kmeans algo on the numpy array and return corresponding labels
    labels, ls_cluster = cluster('AffinityPropagation', X)
    # append the labels to the original dataframe, so we get a dataframe with original features and clusters
    df['label'] = labels.tolist()

    # plotting
    plot('AffinityPropagation', df, ls_cluster)
