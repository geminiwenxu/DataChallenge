import time
import trimap
import umap
import umap.umap_ as umap
import pacmap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import streamlit as st


def dr(path):
    df = pd.read_csv(path, sep=',')
    df = df.fillna(0)
    X = df[df.columns[:-1]].to_numpy()
    y = df['SepsisLabel'].to_numpy()
    ###### only for task 3
    # SepsisLabel = df['SepsisLabel']
    ######

    algorithms = {
        't-SNE': TSNE(),
        'UMAP': umap.UMAP(),
        'TriMAP': trimap.TRIMAP(),
        'PaCMAP': pacmap.PaCMAP(n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
    }

    embeddings = {}
    durations = {}

    for name, algorithm in algorithms.items():
        t0 = time.time()
        if name == "PaCMAP":
            embedding = algorithm.fit_transform(X, init='pca')
            ##### for task 3 generate dim reduced csv
            # df = pd.DataFrame(embedding, columns=['x', 'y'])
            # df = pd.concat([df, SepsisLabel], axis=1)
            #####
            embeddings[name] = pd.DataFrame(embedding, columns=['x', 'y'])
            durations[name] = time.time() - t0
        else:
            embedding = algorithm.fit_transform(X)
            embeddings[name] = pd.DataFrame(embedding, columns=['x', 'y'])
            durations[name] = time.time() - t0
    ###### only for task 3
    # df.to_csv('dr_age_sub_20_40.csv', index=False)
    #######


    fig, ax = plt.subplots(4, figsize=(6, 6))
    for i, j in enumerate(embeddings):
        ax[i].scatter(embeddings[j]['x'], embeddings[j]['y'], cmap="YlOrBr_r", c=y, s=0.6)
        ax[i].set_title(f"Dimentionality reduced by {j}")
    fig.tight_layout()
    # st.pyplot(fig)
    plt.show()


if __name__ == '__main__':
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/gender/Gender0")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/gender/Gender1")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/imputation/mean.csv")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/age/40<Age<70.csv")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/age/Age>70.csv")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/imputation/meansetA.csv")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/imputation/mediansetA.csv")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/imputation/modesetA.csv")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/imputation/linearsetA.csv")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/imputed_subgroup/20<linear.csv")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/imputed_subgroup/20<linear<40.csv")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/imputed_subgroup/40<linear<70.csv")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/imputed_subgroup/70<linear.csv")
    dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/imputed_subgroup/20<mean<40.csv")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/imputed_subgroup/40<mean<70.csv")
    # dr(path="/Users/wenxu/PycharmProjects/DataChallenge/data/imputed_subgroup/70<mean.csv")