
import bhtsne
import umap
import pandas as pd 
import numpy as np 
from sklearn.cluster import MiniBatchKMeans

def tsne(df:pd.DataFrame, dim:int =2, seed:int=0):
    emb = bhtsne.tsne(df.astype(np.float64), dimensions=dim, seed=seed)
    return emb


def umap(df:pd.DataFrame, dim:int = 2, seed:int=0):
    um = umap.UMAP(n_components=dim, random_state=seed)
    um.fit(df)
    return um.transform(df)


def kmeans(df:pd.DataFrame, n_clusters, seed=0):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed)
    kmeans.fit(df)
    return kmeans.transform(df)