from shapely.geometry import box, MultiPolygon, Polygon
from shapely.ops import unary_union
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from typing import List
import utilities as util
import config as conf


def geo_names() -> pd.DataFrame:
    return pd.read_pickle(os.path.join(conf.support_path, 'map-geo-names-DataFrame.pkl'))


def map_geometry() -> MultiPolygon:
    whole_map = pd.read_pickle(os.path.join(conf.support_path, 'map-geometry-MultiPolygon.pkl'))
    datum, stats = util.simplify_multi_poly(whole_map, None)
    print(json.dumps(stats, indent=2))
    return datum


def eco_regions(masked: bool = True) -> Polygon or gpd.GeoDataFrame:
    gdf = pd.read_pickle(os.path.join(conf.support_path, 'map-eco-regions-GeoDataFrame.pkl'))
    region_polys = [e.geometry for j, e in gdf.iterrows()]
    return gdf if masked is False else unary_union(region_polys)


def protected_areas() -> pd.DataFrame:
    return pd.read_pickle(os.path.join(conf.support_path, 'map-protected-areas-DataFrame.pkl'))


def depth_dict() -> dict:
    src = os.path.join(conf.support_path, 'map-depth-points-dict.json')
    with open(src) as depth_file:
        deep_dict = json.load(depth_file)
    deep_dict['lats'] = np.array(deep_dict['lats'])
    deep_dict['lons'] = np.array(deep_dict['lons'])
    deep_dict['data'] = np.array(deep_dict['data'])
    return deep_dict


def depth_contours() -> List:
    return pd.read_pickle(os.path.join(conf.support_path, 'map-depth-contours-list.pkl'))


def map_layers() -> List:
    return pd.read_pickle(os.path.join(conf.support_path, 'map-sector-layers-list.pkl'))


def places() -> pd.DataFrame:
    return pd.read_pickle(os.path.join(conf.support_path, 'map-places-DataFrame.pkl'))


def geo_associations() -> pd.DataFrame:
    return pd.read_pickle(os.path.join(conf.support_path, 'map-geo-associations-DataFrame.pkl'))

