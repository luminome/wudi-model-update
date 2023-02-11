#!/usr/bin/env python3
import os
import config as conf
import json
from shapely.geometry import box
from pathlib import Path

from typing import List
import pandas as pd
import numpy as np
import utilities as util

from builder import load
from tools import get_sector_depth_map


class Sector:
    def __init__(self, s_id, bounding_box, loc, degree_scale):
        self.s_id = s_id
        self.box = bounding_box
        self.path = f"Sector-{s_id}"
        self.degree_scale = degree_scale
        self.data_layers = {}
        self.loc = loc
        self.meta = {}
        self.get_meta()

    def get_meta(self):
        s_path = os.path.join(conf.static_data_path, f"deg_{str(self.degree_scale).replace('.', '_')}", self.path)
        Path(s_path).mkdir(parents=True, exist_ok=True)
        meta_asset_path = os.path.join(s_path, "meta.json")
        has_meta = os.path.exists(meta_asset_path)

        if has_meta:
            with open(meta_asset_path, 'r+') as fp:
                self.meta = json.load(fp)
        else:
            with open(meta_asset_path, 'w+') as fp:
                json.dump({}, fp, indent=2)

    def save(self):
        s_path = os.path.join(conf.static_data_path, f"deg_{str(self.degree_scale).replace('.', '_')}", self.path)
        Path(s_path).mkdir(parents=True, exist_ok=True)
        bytes_saved = 0

        for k, v in self.data_layers.items():
            if k not in self.meta.keys():
                self.meta[k] = []

            for nk, nv in v.items():
                if any(util.flatten_list(nv)):
                    self.meta[k].append(int(nk))
                    sector_asset_path = os.path.join(s_path, f"{k}-{nk}.json")
                    with open(sector_asset_path, 'w') as fp:
                        json.dump(nv, fp)
                    bytes_saved += os.path.getsize(sector_asset_path)

            self.meta[k] = list(dict.fromkeys(self.meta[k]))

        meta_asset_path = os.path.join(s_path, "meta.json")
        with open(meta_asset_path, 'w') as fp:
            json.dump(self.meta, fp, indent=2)

        print(self.path, 'saved', f"{bytes_saved/1000}k")

    def add_data(self, level: int, label: str, flat_coords: list):
        if label not in self.data_layers.keys():
            self.data_layers[label] = {}
        self.data_layers[label][str(level)] = flat_coords


def init_sector(num, m_wid, deg_scale, m_bnd) -> Sector:
    y = np.floor(num / m_wid)
    x = num % m_wid
    sector_box = box(
        m_bnd[0] + x * deg_scale,
        m_bnd[3] - y * deg_scale,
        m_bnd[0] + x * deg_scale + deg_scale,
        m_bnd[3] - y * deg_scale - deg_scale)
    sector_tuple = (m_bnd[0] + (x * deg_scale), m_bnd[3] - (y * deg_scale),)
    return Sector(num, sector_box, sector_tuple, deg_scale)


def build_sectors(map_deg_scale: int) -> List:

    width = np.ceil(conf.master_bounds[1] * (1 / map_deg_scale))
    height = np.ceil(conf.master_bounds[2] * (1 / map_deg_scale))
    sector_count = width * height

    print("sector raw dimensions", width, 'lon x', height, 'lat')
    print("sector instance count", sector_count)
    return [init_sector(n, width, map_deg_scale, conf.master_bounds[0]) for n in range(int(sector_count))]


def get_sector_protected_areas(mpa: pd.DataFrame, sector: Sector, level: int) -> List:
    s_range = conf.levels_range - 1
    simp_limits = conf.mpa_simp_limits
    local_simplification_list = np.linspace(simp_limits[0], simp_limits[1], s_range)

    bnd = sector.box.bounds
    sel_mpa = mpa[((mpa['LON'] > bnd[0]) & (mpa['LON'] < bnd[2])) & ((mpa['LAT'] > bnd[1]) & (mpa['LAT'] < bnd[3]))]
    assoc_mpa = []
    rep_count = 0
    for nj, g in sel_mpa.iterrows():
        if level == conf.levels_range - 1:
            assoc_mpa.append({'id': g.name, 'line_strings': util.geometry_to_coords(g['geometry'])})
        else:
            gk = g['geometry'].simplify(local_simplification_list[level])
            assoc_mpa.append({'id': g.name, 'sub_id': rep_count, 'line_strings': util.geometry_to_coords(gk)})

        rep_count += 1

    return assoc_mpa


def save_sectors():
    for deg in conf.master_degree_intervals:
        sector_group = build_sectors(deg)  #aka '2'
        map_levels = load.map_layers()
        protected_areas = load.protected_areas()
        depth_source = load.depth_dict()

        for level, geometry in enumerate(map_levels):
            print('map level:', level)

            for i, sector in enumerate(sector_group):

                relevant_indices = [r for (r, k) in enumerate(geometry['polygons']) if k.intersects(sector.box)]

                if 'map_polygons' in conf.sector_save_layers:
                    polys = [sector.box.intersection(geometry['polygons'][p]) for p in relevant_indices]
                    sector.add_data(level, 'polygons', [util.geometry_to_coords(fp) for fp in polys])

                if 'map_lines' in conf.sector_save_layers:
                    lines = [sector.box.intersection(geometry['line_strings'][p]) for p in relevant_indices]
                    sector.add_data(level, 'line_strings', [util.geometry_to_coords(fp) for fp in lines])

                if 'depth_contours' in conf.sector_save_layers:
                    contour_set = []
                    for contour_depth in geometry['contours']:
                        contour_indices = [r for (r, k) in enumerate(contour_depth['contours']) if k.intersects(sector.box)]
                        contours = [sector.box.intersection(contour_depth['contours'][p]) for p in contour_indices]
                        contour_labels_indices = [r for (r, k) in enumerate(contour_depth['labels']) if k.intersects(sector.box)]
                        contour_labels = [sector.box.intersection(contour_depth['labels'][p]) for p in contour_labels_indices]
                        if len(contours):
                            contour_set.append({
                                'd': contour_depth['d'],
                                'line_strings': [util.geometry_to_coords(fp) for fp in contours],
                                'labels': [util.geometry_to_coords(fp) for fp in contour_labels]
                            })

                    sector.add_data(level, 'contours', contour_set)

                if 'depth_maps' in conf.sector_save_layers:
                    packet = get_sector_depth_map(depth_source, geometry['contours'], sector, level)
                    if packet['contour_lines'] is not None:
                        sector.add_data(level, 'depth_maps', [packet])

                if 'protected_areas' in conf.sector_save_layers:
                    sector.add_data(level, 'protected_areas', get_sector_protected_areas(protected_areas, sector, level))

                util.show_progress('sectors', i, len(sector_group))

        # important
        [sector.save() for sector in sector_group]




