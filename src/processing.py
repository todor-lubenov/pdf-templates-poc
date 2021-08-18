import datetime
import json
import os
import time
import numpy as np
import pandas as pd
from skimage.exposure import histogram
from skimage import io
from skimage.measure import shannon_entropy
import pickle


def get_thumbnails(arr):
    for el in arr:
        files = list(next(os.walk(f"{base_uri}/{el['dpath']}/page_thumbnails"))[2])
        el['img_files_thumbnails'] = files


def get_num_pages(arr):
    for e in arr:
        e['imagery']['num_pages'] = len(e['img_files_thumbnails'])


def get_min_max(arr):
    mina, maxa = 10, 0

    for e in templs:
        if e['imagery']['num_pages'] > maxa:
            maxa = e['imagery']['num_pages']
        if e['imagery']['num_pages'] < mina:
            mina = e['imagery']['num_pages']

    return mina, maxa


def get_oner_page(arr):
    for e in arr:
        if e['imagery']['num_pages'] == 1:
            print(e)


def sort_thumbnails(arr):
    for e in arr:
        e['img_files_thumbnails'].sort()


def augment_objects(arr):
    s = time.perf_counter()

    for e in arr:
        f_page = e['img_files_thumbnails'][0]
        first_page = f"{base_uri}/{e['dpath']}/page_thumbnails/{f_page}"
        first_page_image = io.imread(first_page)
        hist, hist_centers = histogram(first_page_image)
        e['first_page_height'], e['first_page_width'], e['first_page_layers'] = first_page_image.shape
        e['first_page_hist'] = hist
        e['first_page_hist_centers'] = hist_centers
        e['shannon_entropy_2'] = shannon_entropy(first_page_image, base=2)
        e['img_mean'] = np.mean(first_page_image)
        e['img_median'] = np.median(first_page_image)
        e['img_std'] = np.std(first_page_image)
        e['img_variance'] = np.var(first_page_image)
        e['img_average'] = np.average(first_page_image)

    print(f'image meta fetch costs ~ {time.perf_counter() - s} seconds')


def get_df(arr):
    width_arr = []
    height_arr = []
    shannon_arr = []
    mean_arr = []
    median_arr = []
    std_arr = []
    var_arr = []

    for e in arr:
        width_arr.append(e['first_page_width'])
        height_arr.append(e['first_page_height'])
        shannon_arr.append(e['shannon_entropy_2'])
        mean_arr.append(e['img_mean'])
        median_arr.append(e['img_median'])
        std_arr.append(e['img_std'])
        var_arr.append(e['img_variance'])
    df = pd.DataFrame({'img_mean': mean_arr,
                       'img_median': median_arr,
                       'img_shannon_2': shannon_arr,
                       'img_std': std_arr,
                       'img_var': var_arr,
                       'img_width': width_arr,
                       'img_height': height_arr})
    return df


if __name__ == '__main__':
    # where normalized data/struct has been stored after preprocessing
    base_uri = '/Users/todorlubenov/Documents/AllianzUK/'

    with open(f'{base_uri}meta_data.json') as outfile:
        templs = json.load(outfile)

    get_thumbnails(templs)
    get_num_pages(templs)
    sort_thumbnails(templs)
    augment_objects(templs)

    now = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')

    with open(f'{base_uri}/templates_features_{now}.pickle', 'wb') as handle:
        pickle.dump(templs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df = get_df(templs)
    df.to_parquet(f'{base_uri}classification_df_{now}.pq')

