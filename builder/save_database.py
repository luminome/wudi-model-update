#!/usr/bin/env python3

import config as conf
from builder import load
from updater import db_manage
import utilities as util


def places():
    df = load.places()
    mods = [['lon', 'number'], ['lat', 'number']]
    for m in mods:
        df[m[0]] = df[m[0]].apply(util.db_value_cleaner, args=(conf.db_float_precision, m[1],))

    dtypes = {
        'lon': 'REAL',
        'lat': 'REAL',
        'region': 'INT',
        'geo': 'INT',
        'wudi': 'INT',
        'name': 'TEXT',
        'townLabel': 'TEXT',
        'countryLabel': 'TEXT',
        'area': 'REAL',
        'population': 'INT',
        'elevation': 'REAL',
        'regionLabels': 'TEXT',
        'waterLabels': 'TEXT',
        'capital': 'TEXT',
        'type': 'TEXT',
        'place_id': 'INT',
        'source': 'TEXT',
        'status': 'INT',
    }

    conn = db_manage.create_connection(conf.wudi_model_database_path)
    df.to_sql('places', conn, if_exists='replace', dtype=dtypes, index=False)
    conn.close()


def protected_areas():
    df = load.protected_areas()
    df = df.drop(columns=['geometry', 'LON', 'LAT'])
    conn = db_manage.create_connection(conf.wudi_model_database_path)
    df.to_sql('protected_areas', conn, if_exists='replace', index=False)
    conn.close()


def geo_associations():
    df = load.geo_associations()

    mods = [['pid', 'number'], ['within_protected', 'number']]
    for m in mods:
        df[m[0]] = df[m[0]].apply(util.db_value_cleaner, args=(conf.db_float_precision, m[1],))

    dtypes = {
        'pid': 'INT',
        'unique_protected': 'INT',
        'nearest_protected': 'INT',
        'within_protected': 'TEXT',
        'unique_place': 'INT',
        'nearest_place': 'INT',
        'country': 'TEXT'
    }

    conn = db_manage.create_connection(conf.wudi_model_database_path)
    df.to_sql('geo_associations', conn, if_exists='replace', dtype=dtypes, index=False)
    conn.close()
    pass


def tests():
    places()
    protected_areas()
    geo_associations()