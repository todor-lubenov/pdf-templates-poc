
# !pip install scikit-image
# !pip install pdfminer.six

from typing import List, Dict
import pandas as pd
import time
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
import numpy as np


def get_x(d: pd.DataFrame) -> pd.DataFrame:
    r = d[[#'form_size_kb',
           #'form_num_pages',
           'page_height',
           'page_width',
           #'page_layers',
           'img_shannon_2',
           'img_mean',
           'img_median',
           'img_std',
           'img_variance']]
    return r


def get_knn(x: pd.DataFrame, n_clusters: int) -> List:
    kmeans = KMeans(n_clusters=n_clusters).fit(x)
    return kmeans.labels_


def get_spectral_clustering(x: pd.DataFrame, n_clusters: int) -> List:
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize').fit(x)
    return clustering.labels_


def get_agglomerative_clustering(x: pd.DataFrame) -> List:
    clustering = AgglomerativeClustering().fit(x)
    return clustering.labels_


if __name__ == '__main__':
    s = time.perf_counter()
    try:
        from . import base_destination
    except Exception as ex:
        print(ex)
        base_destination = '/Users/todorlubenov/Documents/AllianzUK/'

    with open(f'nower', 'r') as f:
        now = f.readline()

    df = pd.read_parquet(f'{base_destination}classification_df_{now}.pq')
    print(df.info())
    print(df.head())

    X = get_x(df)

    n_clusters = 25

    df[f'knn_{n_clusters}'] = get_knn(X, n_clusters)
    df[f'spectral_{n_clusters}'] = get_spectral_clustering(X, n_clusters)
    df['agglomerative'] = get_agglomerative_clustering(X)

    df.to_parquet(f'{base_destination}experiment_clust_{n_clusters}_{now}.pq')

    print(f'total exec time is ~ {time.perf_counter() - s} seconds')

