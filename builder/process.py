#!/usr/bin/env python3

import os
import config as conf
import utilities as util
from typing import List


"""
places_dataframe and protected_areas_dataframe:
post-processing of raw data.
"""


def places_dataframe() -> List:
    import pandas as pd
    df_list = pd.read_pickle(os.path.join(conf.support_path, 'map-places-overpass-wiki-list.pkl'))
    df = pd.DataFrame([t for t in df_list])
    status_field = [None] * len(df_list)
    df['status'] = status_field

    df = df.astype({
        'population': 'float64',
        'lon': 'float64',
        'lat': 'float64',
    })

    df = df.drop(df[(df.population < 1000) | (df.population.isnull())].index)
    print(df.info())

    return [util.save_asset(df, 'map-places')]


def protected_areas_dataframe() -> List:
    import pandas as pd
    df = pd.read_pickle(os.path.join(conf.support_path, 'map-protected-areas-raw-DataFrame.pkl'))
    print(df.info())

    mods = [['CENTROID', 'number', 'centroid'], ['COUNTRY', 'string', 'COUNTRY'], ['MED_REGION', 'string', 'MED_REGION']]
    for m in mods:
        df[m[0]] = df[m[0]].apply(util.db_value_cleaner, args=(conf.db_float_precision, m[1],))

    df.NAME = df.NAME.apply(util.title_case)

    return [util.save_asset(df, 'map-protected-areas')]


def post():
    print(places_dataframe())
    print(protected_areas_dataframe())
