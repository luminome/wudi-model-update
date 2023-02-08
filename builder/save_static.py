#!/usr/bin/env python3

import os
import config as conf
import utilities as util
import pandas as pd


def iso_bath(name: str, data: list):
    contours = [util.value_cleaner(x) for x in data[0]['contours']]
    filtered = [c for c in contours if c != '[[]]']
    content = '"{:s}",{:s}'.format(name, ','.join(filtered)) + ','

    file_name = f"{name}-{int(abs(conf.contour_ranges['iso_bath_depth']))}-1.txt"
    iso_bath_path = os.path.join(conf.static_data_path, file_name)

    with open(iso_bath_path, "w+") as iso_bath_file:
        iso_bath_file.write(content[:-1])


def data_to_text(support_file: str):
    file = conf.support_files[support_file]
    df = pd.read_pickle(os.path.join(conf.support_path, file))
    print(df.__class__)

    if df.__class__ == list and support_file == 'map-iso-bath':
        iso_bath(support_file, df)
    else:
        print(df.info())

        cols = list(df)
        col_count = len(cols) + 1

        raw = '"ID",' + ','.join([util.value_cleaner(x) for x in cols]) + ','

        for j, e in df.iterrows():
            row_str = str(j) + ',{:s}'.format(','.join([util.value_cleaner(x, conf.db_float_precision, True) for x in e])) + ','
            raw += row_str

        file_name = f"{support_file}-{col_count}.txt"
        path = os.path.join(conf.static_data_path, file_name)
        with open(path, "w+") as file:
            file.write(raw[:-1])


def tests():
    pass
    # as_text('map-iso-bath')
    # as_text('map-geo-names')
