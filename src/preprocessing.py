
# !pip install pdf2image
# !pip install --upgrade pip
# !pip install joblib

from typing import Dict, List

import os
import pathlib
import shutil
from datetime import datetime
import time
import json

from pdf2image import convert_from_path, convert_from_bytes

from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


def generate_images_from_pdf(srs_pdf_path: str, dest_folder: str, props={}):
    convert_from_path(srs_pdf_path,
                      dpi=props.get('DPI', 200),
                      output_folder=dest_folder,
                      grayscale=True,
                      output_file=props.get('fname', 'cytora_init'),
                      fmt="png"
                      )
    form_meta = {
        'num_pages': 0,
        'pages_extent': {},
        'pages_hist': {},
        'pages_': {}
    }
    return form_meta


def filename_normalizer(fname: str):
    tmp = fname.lower()
    tmp = tmp.replace('.pdf', '')
    tmp = tmp.replace('(', '_').replace('__', '_')
    tmp = tmp.replace(')', '_').replace('__', '_')
    tmp = tmp.replace(' ', '_').replace('__', '_')
    tmp = tmp.replace('.', '_').replace('__', '_')
    tmp = tmp.replace('-', '_').replace('__', '_')

    tmp = tmp.replace('__', '_').replace('__', '_').replace('__', '_')
    tmp = tmp.strip('_')

    return tmp


def create_form_folder(dest_folder: str, form_object: Dict):
    s = os.getcwd()
    print(s)
    now = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    dpath = f"{form_object['file_name_normalized']}_{now}"
    if form_object.get('abs_path', None):
        if not os.path.exists(f'{dest_folder}/{dpath}'):
            os.makedirs(f'{dest_folder}/{dpath}')
        os.chdir(f'{dest_folder}/{dpath}')

        if not os.path.exists('source_form'):
            os.makedirs('source_form')
        # copy source file to dest folder in dedicated subfolder "source_form"

        # create folder for store converted images
        if not os.path.exists('page_images'):
            os.makedirs('page_images')

        # create folder for thumbnails
        if not os.path.exists('page_thumbnails'):
            os.makedirs('page_thumbnails')

        # create folder for experiments
        if not os.path.exists('experiments'):
            os.makedirs('experiments')

        # create folder for features
        if not os.path.exists('features'):
            os.makedirs('features')

        # make a copy of the source form to destination form folder
        src = form_object['abs_path']
        dst = f"{dest_folder}/{dpath}/source_form/{form_object['file_name_normalized']}{form_object['extension']}"
        shutil.copy(src, dst)
    os.chdir(s)
    return dpath


def get_pdf_files_list_of_objects(base_uri: str) -> List:
    exts = {}
    pdf_files = []

    for root, folder, files in os.walk(base_uri):
        for file in files:
            ext = pathlib.Path(file).suffix
            if exts.get(ext, None):
                exts[ext] += 1
            else:
                exts[ext] = 1
            if pathlib.Path(file).suffix.lower() == '.pdf':
                obj = {
                    'file_name': pathlib.Path(file).stem,
                    'extension': '.pdf',
                    'abs_path': os.path.join(root, file),
                    'st_size_bytes': pathlib.Path(os.path.join(root, file)).stat().st_size,
                }
                pdf_files.append(obj)
    return pdf_files


if __name__ == '__main__':
    from . import base_uri, base_destination

    pdf_files = get_pdf_files_list_of_objects(base_uri)
    for el in pdf_files:
        el['file_name_normalized'] = filename_normalizer(el['file_name'])

    assert len(pdf_files) == 687

    for el in pdf_files:
        print(el)
        time.sleep(1)
        el['dpath'] = create_form_folder(base_destination, el)

    for el in pdf_files:
        try:
            el['imagery'] = generate_images_from_pdf(
                f"{base_destination}{el['dpath']}/source_form/{el['file_name_normalized']}{el['extension']}",
                f"{base_destination}{el['dpath']}/page_thumbnails", {'DPI': 50})
        except Exception as ex:
            el['error'] = str(ex)

    with open(f'{base_destination}meta_data.json', 'w') as outfile:
        json.dump(pdf_files, outfile)
