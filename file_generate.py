import pacmap as pacmap
import pandas as pd
import pacmap
import trimap
import umap
import umap.umap_ as umap
from sklearn.manifold import TSNE

if __name__ == '__main__':
    df = pd.read_csv("//data/age/20<Age<40.csv", sep=',')
    df = df.fillna(0)
    # print(df[df.columns[:8]])
    selected_feature_df = pd.concat([df[df.columns[:7]], df['SepsisLabel']], axis=1)
    selected_feature_df.to_csv('selected_feature_20_40.csv', index=False)
