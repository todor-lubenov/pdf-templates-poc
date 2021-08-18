
from typing import List
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


def get_thumbnails(arr: List, uri: str):
    for el in arr:
        files = list(next(os.walk(f"{uri}/{el['dpath']}/page_thumbnails"))[2])
        el['img_files_thumbnails'] = files


def get_num_pages(arr: List):
    for e in arr:
        e['imagery']['num_pages'] = len(e['img_files_thumbnails'])


def get_min_max(arr: List):
    mina, maxa = 10, 0
    sz = 0

    for e in arr:
        if e['imagery']['num_pages'] > maxa:
            maxa = e['imagery']['num_pages']
        if e['imagery']['num_pages'] < mina:
            mina = e['imagery']['num_pages']
        if e['st_size_bytes'] > sz:
            sz = e['st_size_bytes']

    return mina, maxa, sz


def get_oner_page(arr: List):
    for e in arr:
        if e['imagery']['num_pages'] == 1:
            print(e)


def sort_thumbnails(arr: List):
    for e in arr:
        e['img_files_thumbnails'].sort()


def augment_objects(arr: List, uri: str):
    s = time.perf_counter()

    for e in arr:
        f_page = e['img_files_thumbnails'][0]
        first_page_image = io.imread(f"{uri}/{e['dpath']}/page_thumbnails/{f_page}")
        e['first_page_height'], e['first_page_width'], e['first_page_layers'] = first_page_image.shape
        e['first_page_hist'], e['first_page_hist_centers'] = histogram(first_page_image)
        e['shannon_entropy_2'] = shannon_entropy(first_page_image, base=2)
        e['img_mean'] = np.mean(first_page_image)
        e['img_median'] = np.median(first_page_image)
        e['img_std'] = np.std(first_page_image)
        e['img_variance'] = np.var(first_page_image)
        e['img_average'] = np.average(first_page_image)

    print(f'image meta fetch costs ~ {time.perf_counter() - s} seconds')


def get_img_df(arr: List, uri: str) -> pd.DataFrame:
    res = []

    for e in arr:
        obj = {
            'form_path': f"{uri}{e['dpath']}/source_form/{e['file_name_normalized']}",
            'form_size_kb': e['st_size_bytes']/1024,
            'form_num_pages': e['imagery']['num_pages'],
            'page_height': e['first_page_height'],
            'page_width': e['first_page_width'],
            'page_layers': e['first_page_layers'],
            'img_shannon_2': e['shannon_entropy_2'],
            'img_mean': e['img_mean'],
            'img_median': e['img_median'],
            'img_std': e['img_std'],
            'img_variance': e['img_variance']
        }
        res.append(obj)

    res_df = pd.DataFrame(res)
    return res_df


def get_df(arr: List):
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
    d = pd.DataFrame({'img_mean': mean_arr,
                      'img_median': median_arr,
                      'img_shannon_2': shannon_arr,
                      'img_std': std_arr,
                      'img_var': var_arr,
                      'img_width': width_arr,
                      'img_height': height_arr}
                     )
    return d


if __name__ == '__main__':
    s = time.perf_counter()
    try:
        from . import base_destination
    except Exception as ex:
        print(ex)
        base_destination = '/Users/todorlubenov/Documents/AllianzUK/'

    with open(f'{base_destination}meta_data.json') as outfile:
        templs = json.load(outfile)

    get_thumbnails(templs, base_destination)
    get_num_pages(templs)
    sort_thumbnails(templs)
    augment_objects(templs, base_destination)

    now = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    with open(f'nower', 'w') as f:
        f.write(now)

    with open(f'{base_destination}templates_features_{now}.pickle', 'wb') as handle:
        pickle.dump(templs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # df = get_df(templs)
    df = get_img_df(templs, base_destination)
    df.to_parquet(f'{base_destination}classification_df_{now}.pq')
    print(f'total exec time is ~ {time.perf_counter() - s} seconds')
