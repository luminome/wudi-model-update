#!/usr/bin/env python3

import os
import json
import math
import numpy as np
import config as conf
from typing import List
import itertools
from decimal import *
from shapely.geometry import Point, Polygon, MultiPolygon, LinearRing, LineString, MultiLineString

mod_trace = {}
formatter = {'set': ''}


class JsonSafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, np.nan):
            return '"'+str(obj)+'"'
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 5)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, MultiPolygon):
            return str(obj.__class__)
        if isinstance(obj, Polygon):
            return str(obj.__class__)
        if isinstance(obj, LineString):
            return str(obj.__class__)
        if isinstance(obj, MultiLineString):
            return str(obj.__class__)
        if isinstance(obj, LinearRing):
            return str(obj.__class__)

        return super(JsonSafeEncoder, self).default(obj)

#
# class JsonDecimalEncoder(json.JSONEncoder):
#     #alphonso baschiera
#     def default(self, obj):
#         # 👇️ if passed in object is instance of Decimal
#         # convert it to a string
#         if isinstance(obj, Decimal):
#             return str(obj)
#         # 👇️ otherwise use the default behavior
#         return json.JSONEncoder.default(self, obj)


def title_case(string):
    return string.title()


def flatten_list(element):
    return sum(map(flatten_list, element), []) if isinstance(element, list) else [element]
    # see notes.


def flatten_coords(coords, dec):
    arr = [[round(c[0], dec), round(c[1], dec)] for c in coords]
    return list(itertools.chain(*arr))
    #return seq[0] if len(seq) else seq


def cleaner_numeric(obj, precision):
    c = obj.__class__
    # print(c, str(obj))
    if c == int:
        return int(obj)
    elif obj.is_integer():
        return int(obj)
    elif np.isnan(obj):
        return None
    elif c == float:
        return round(obj, precision)
    else:
        return obj


def save_asset(asset, file_name, alt_path=None):
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import pickle
    import config as conf
    from shapely.geometry import MultiPolygon
    from datetime import datetime
    #from updater.json_encoder import JsonSafeEncoder

    class_name = type(asset).__name__.split('.')[-1]
    print(class_name, type(asset))
    asset_path, path, status = None, None, None

    if alt_path is None:
        asset_path = conf.support_path
    else:
        asset_path = alt_path

    if type(asset) in [np.ndarray, np.array]:
        path = os.path.join(asset_path, f"{file_name}-{class_name}.npy")
        np.save(str(path), asset, allow_pickle=True)

    if type(asset) in [np.ma.core.MaskedArray]:
        path = os.path.join(asset_path, f"{file_name}-{class_name}.npy")
        b = {'data': asset}
        np.save(str(path), b, allow_pickle=True)

    if type(asset) in [pd.DataFrame, gpd.GeoDataFrame]:
        path = os.path.join(asset_path, f"{file_name}-{class_name}.pkl")
        asset.to_pickle(path)

    if type(asset) in [list, MultiPolygon]:
        path = os.path.join(asset_path, f"{file_name}-{class_name}.pkl")
        with open(path, "wb") as file:
            pickle.dump(asset, file, pickle.HIGHEST_PROTOCOL)

    if type(asset) is dict:
        path = os.path.join(asset_path, f"{file_name}-{class_name}.json")
        with open(path, "w") as file:
            json.dump(asset, file, cls=JsonSafeEncoder)  #, indent=2,

    if path is not None:
        status = f"{str(datetime.now())}\n{path} {os.path.getsize(path)/1000}k\n"
        with open(os.path.join(asset_path, 'history.txt'), 'a+') as history:
            history.write(status)

    print(status)


def show_progress(item_name, count, total):
    if total > 0:
        pit = math.ceil((count / (total-1)) * 100)
    else:
        pit = 'NaN'

    if count == total-1:
        print(f"\r{item_name} completed", end='')
        print('')
    else:
        print(f"\r{item_name} {pit}% complete", end='')


def simplify_multi_poly(source_poly, f_range=None) -> tuple:
    from shapely.geometry import MultiPolygon
    #// simplest -> most complex
    s_range = conf.levels_range-1  # if f_range is None else f_range-1

    area_limits = conf.area_limits
    simp_limits = conf.simp_limits

    simplifications = np.linspace(area_limits[0], area_limits[1], s_range)
    area_limits = np.linspace(simp_limits[0], simp_limits[1], s_range)
    mod_trace['simplify_multi_poly'] = {'simplifications': list(simplifications), 'area_limits': list(area_limits)}

    poly_levels = []

    for b in range(s_range):
        this_map = []
        for i, poly in enumerate(source_poly.geoms):
            simp_poly = poly.simplify(simplifications[b])
            if simp_poly.area >= area_limits[b]:
                this_map.append(simp_poly)

        poly_levels.append(MultiPolygon(this_map))

        if s_range > 1:
            show_progress('simplify_multi_poly', b, s_range)

    poly_levels.append(source_poly)

    for i, source_multi_poly in enumerate(poly_levels):
        mod_trace['simplify_multi_poly'][f'result-{i}'] = [
            f'{len(source_multi_poly.geoms)} polygons',
            f'{sum([len(p.exterior.coords) for p in source_multi_poly.geoms])} coordinates'
        ]

    return poly_levels, mod_trace


def poly_s_to_list(reso, min_length=0.0) -> List:
    result = []
    if reso.type == 'MultiPolygon' or reso.type == 'MultiLineString':
        result = [line for line in reso.geoms if line.length > min_length]
    elif reso.type == 'Polygon' or reso.type == 'LineString':
        if reso.length > min_length:
            result.append(reso)
    return result


def geometry_to_coords(geom, decimal_places=4):
    def getter(element):
        if element.type in ['Polygon']:
            return flatten_coords(element.exterior.coords, decimal_places)
        if element.type in ['LineString']:
            return flatten_coords(element.coords, decimal_places)

    if geom.type == 'MultiPolygon' or geom.type == 'MultiLineString':
        return [getter(element) for element in geom.geoms]
    else:
        return [getter(geom)]


def polygon_to_serialized_obj(poly, decimal_places):
    out = flatten_coords(poly.exterior.coords, decimal_places)
    ins = [flatten_coords(interior.coords, decimal_places) for interior in poly.interiors]
    return {'out': out, 'in': ins}


def get_data_scale_extents(data_dict):
    lo = data_dict['lons']
    la = data_dict['lats']
    return lo[0], la[0]


def normalize_val(val, mi, ma):
    return (val - mi) / (ma - mi)


def show_support_file(args):
    import pandas as pd

    kext = str(args[0]).split('.')[-1]
    path = os.path.join(conf.support_path, str(args[0]))

    if kext == 'npy':
        data = np.load(path, None, True)
        print(data.shape)
        print(data[0])

    if kext == 'pkl':
        df = pd.read_pickle(path)
        print(df.__class__)

        if df.__class__ == list:
            for i, tem in enumerate(df):
                print(i, tem)
        else:
            print(df.info())
            for n in range(df.shape[0]):
                print(list(df.iloc[n]))


def db_value_floated(obj, fmt):
    #print(obj.__class__.__name__)
    if obj.__class__ == int:
        return str(obj)
    elif obj.is_integer():
        return str(int(obj))
    elif np.isnan(obj):
        return None
    else:
        return fmt.format(obj)


def db_value_cleaner(obj, precision, o_type):
    obj_cls = obj.__class__
    formatter['set'] = '{:.%if}' % precision
    fmt = formatter['set']
    # print(obj.__class__.__name__)

    if obj_cls == Point:
        cale = [round(obj.x, precision), round(obj.y, precision)]
        return ",".join([fmt.format(c) for c in cale])

    if obj_cls == list:
        if o_type == 'number':
            return ",".join([db_value_floated(c, fmt) for c in obj])
        if o_type == 'string':
            return ",".join(obj)
            pass

    if obj_cls == np.ndarray:
        return ",".join([db_value_floated(c, fmt) for c in obj.tolist()])

    if obj_cls == float:
        return db_value_floated(obj, fmt)

    if obj_cls == int:
        return obj
    #return obj.__class__.__name__+' '+str(arg)


def value_floated(obj, fmt):
    if obj.__class__ == int:
        return obj
    elif obj.is_integer():
        return int(obj)
    elif np.isnan(obj):
        return 'null'
    else:
        return float(fmt.format(obj))


def value_as_string(s):
    return '\"{:s}\"'.format(s)


def value_from_list(s, fmt=None):
    if fmt is None:
        fmt = formatter['set']
    if s.__class__ == str:
        return s  #value_as_string(s)
    elif s.__class__ == float or s.__class__ == int:
        return value_floated(s, fmt)


def value_in_place(s):
    return list(map(value_in_place, s)) if isinstance(s, list) else value_from_list(s)


# VALUE CLEANER FOR TEXT-ASSETS
def value_cleaner(obj, decimal_places=4, special_poly=None):
    # global format
    obj_cls = obj.__class__
    obj_str = None
    f_fmt = '{:.%if}' % decimal_places
    formatter['set'] = f_fmt

    # print(obj_cls)

    if obj_cls == str:
        obj_str = value_as_string(obj)

    if obj_cls == int:
        obj_str = obj

    if obj_cls == float:
        obj_str = value_floated(obj, f_fmt)

    if obj_cls == List or obj_cls == tuple or obj_cls == list:
        obj_str = value_in_place(obj)

    if obj_cls == np.ndarray:
        float_fmt = ['float64', 'float32']
        if obj.dtype in float_fmt:
            obj = np.round(obj, decimal_places)
        obj_str = str(obj.tolist()).replace("'", "\"")

    if obj_cls == MultiPolygon:
        obj_str = value_as_string(obj.__class__.__name__)

    if obj_cls == Polygon and special_poly:
        obj_str = str(polygon_to_serialized_obj(obj, decimal_places)).replace(" ", "")

    if obj_cls == Polygon or obj_cls == LineString or obj_cls == MultiPolygon or obj_cls == MultiLineString:
        obj_str = str(geometry_to_coords(obj, decimal_places)).replace(" ", "")

    if obj_str is None:
        print('no cleaner for', str(obj.__class__))
        return 'null'
    else:
        #print(obj_str)
        return obj_str