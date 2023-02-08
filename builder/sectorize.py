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

                if 'protected_areas' in conf.sector_save_layers:
                    sector.add_data(level, 'protected_areas', get_sector_protected_areas(protected_areas, sector, level))

                util.show_progress('sectors', i, len(sector_group))

        # important
        [sector.save() for sector in sector_group]







#
#
# # this is to make all sectors and all layers
# def make_sectors():
#     print(conf.master_bounds)
#     # save_parsed_map_data(force_range)
#     # exit()
#     s_range = conf.levels_range - 1
#     simp_limits = conf.mpa_simp_limits
#     local_simplification_list = np.linspace(simp_limits[0], simp_limits[1], s_range)
#
#     print(local_simplification_list)
#
#     whole_map_poly_sets = pd.read_pickle(os.path.join(conf.assets_path, 'v2_sector_map_layers-list.pkl'))
#
#     mpa = pd.read_pickle(os.path.join(conf.assets_path, 'v2_protected_regions-DataFrame.pkl'))
#     print(len(whole_map_poly_sets))
#
#     def join_sector_mpa_list(iter_sector, level) -> List:
#         bnd = iter_sector.box.bounds
#         sel_mpa = mpa[((mpa['LON'] > bnd[0]) & (mpa['LON'] < bnd[2])) & ((mpa['LAT'] > bnd[1]) & (mpa['LAT'] < bnd[3]))]
#         assoc_mpa = []
#         rep_count = 0
#         for nj, g in sel_mpa.iterrows():
#             #if g['STATUS_ENG'] == 'Designated':
#             if level == len(whole_map_poly_sets)-1:
#                 # print('add full size')
#                 assoc_mpa.append({'id': g.name, 'line_strings': util.geometry_to_coords(g['geometry'])})
#             else:
#                 # print('simplified geom')
#                 gk = g['geometry'].simplify(local_simplification_list[level])
#                 assoc_mpa.append({'id': g.name, 'sub_id': rep_count, 'line_strings': util.geometry_to_coords(gk)})
#             rep_count += 1
#
#         return assoc_mpa
#
#
#     for deg in conf.master_degree_intervals:
#         sector_group = build_sectors(deg)
#         print("sectors:", len(sector_group))
#
#         for j, geometry in enumerate(whole_map_poly_sets):
#             print('level:', j)
#
#             if j >= force_range:
#                 break
#
#             for i, sector in enumerate(sector_group):
#
#                 mpa_geoms_set = join_sector_mpa_list(sector, j)
#                 relevant_indices = [r for (r, k) in enumerate(geometry['polygons']) if k.intersects(sector.box)]
#                 polys = [sector.box.intersection(geometry['polygons'][p]) for p in relevant_indices]
#                 lines = [sector.box.intersection(geometry['line_strings'][p]) for p in relevant_indices]
#
#                 contour_set = []
#                 for contour_depth in geometry['contours']:
#                     relevant_indices = [r for (r, k) in enumerate(contour_depth['contours']) if k.intersects(sector.box)]
#                     contours = [sector.box.intersection(contour_depth['contours'][p]) for p in relevant_indices]
#                     if len(contours):
#                         contour_set.append({'d': contour_depth['d'], 'line_strings': [util.geometry_to_coords(fp) for fp in contours]})
#
#                 sector.add_data(j, 'polygons', [util.geometry_to_coords(fp) for fp in polys])
#                 sector.add_data(j, 'line_strings', [util.geometry_to_coords(fp) for fp in lines])
#                 sector.add_data(j, 'mpa_s', mpa_geoms_set)
#                 sector.add_data(j, 'contours', contour_set)
#
#                 util.show_progress('sectors', i, len(sector_group))
#
#         #// important
#         [sector.save() for sector in sector_group]
#
#
#
#
#
#
# def prepare_sectors(args):
#     print(args)
#     s_id = int(args[0])
#     s_level = args[1]
#
#     def rounder(v):
#         return round(v, conf.data_decimal_places)
#
#     b_filter = np.vectorize(rounder)
#
#     depth_dict = data_parse.open_depth_file()
#     #//print(depth_dict)
#
#     if s_level == 'all':
#         levels = [0, 1, 2, 3, 4]
#     else:
#         levels = [int(s_level)]
#
#
#     for deg in conf.master_degree_intervals:
#         sector_group = build_sectors(deg)
#         print("sectors:", len(sector_group))
#
#         #for s in sector_group:
#         sector = sector_group[s_id]
#         print(sector.path, sector.loc, sector.box.bounds)
#
#         whole_map_poly_sets = pd.read_pickle(os.path.join(conf.assets_path, 'v2_sector_map_layers-list.pkl'))
#
#         for level in levels:
#             geometry = whole_map_poly_sets[level]
#             batch = data_parse.get_depth_points(depth_dict, sector, level)
#             depth_bounds = box(min(batch[0]), min(batch[1]), max(batch[0]), max(batch[1]))
#
#             print('selected level:', level)
#             print(sector.box)
#             print(depth_bounds)
#
#             relevant_indices = [r for (r, k) in enumerate(geometry['polygons']) if k.intersects(sector.box)]
#             lines = [sector.box.intersection(geometry['line_strings'][p]) for p in relevant_indices]
#
#             # pck_sub = np.delete(batch, np.where((batch[:, 2] > -5))[0], axis=0)
#             # pck_super = np.delete(batch, np.where((batch[:, 2] > 0))[0], axis=0)
#
#             a = batch[:, 0].tolist()
#             b = batch[:, 1].tolist()
#             c = batch[:, 2].tolist()
#
#             for fp in lines:
#                 util.multiline_geometry_to_coord_columns(fp, a, b, c)
#
#             # pck = np.column_stack((a, b, c))
#             # print(pck.shape)
#
#             #pck = np.delete(pck, np.where((pck[:, 0] >= 25) & (pck[:, 0] <= 35))[0], axis=0)
#
#             #
#             # print(a, b, c)
#
#             #fresh_plot_basic(pck_dep[:, 0], pck_dep[:, 1], pck_dep[:, 2])
#             #fresh_plot_basic(np.array(a), np.array(b), np.array(c))
#             #fresh_plot_basic(np.array(a), b, c)
#             #data_parse.get_model([a, b, c])
#
#             index = data_parse.get_mesh_indices([a, b, c])
#             index = index.flatten()
#
#             vert = np.column_stack((a, b, c))
#             vert = vert.flatten()
#
#             vert = b_filter(vert)
#
#             print(index)
#
#             mesh = {
#                 'indices': index.tolist(),
#                 'vertices': vert.tolist()
#             }
#
#             sector.add_data(level, 'meshes', [mesh])
#
#         sector.save()
#
#
#     exit()
#
#
#
#
#         #
#         # whole_map_poly_sets = pd.read_pickle(os.path.join(conf.assets_path, 'v2_sector_map_layers-list.pkl'))
#         #
#         # geometry = whole_map_poly_sets[level]
#         #
#         # print('selected level:', level)
#         #
#         # relevant_indices = [r for (r, k) in enumerate(geometry['polygons']) if k.intersects(sector.box)]
#         #
#         # lines = [sector.box.intersection(geometry['line_strings'][p]) for p in relevant_indices]
#         #
#         # a = []
#         # b = []
#         # c = []
#         # for fp in lines:
#         #     util.multiline_geometry_to_coord_columns(fp, a, b, c)
#         #
#         # #//have columns of points
#         #
#         # sizer = lambda t: (2 + round(abs(t) / 1000))
#         # size_func = np.vectorize(sizer)
#         #
#         # color = lambda t: 'green' if t > 0 else 'blue'
#         # color_func = np.vectorize(color)
#         #
#         # bz = color_func(c)
#         # az = size_func(c)
#         #
#         # d = [a, b]
#         # util.plot_basic(d, az, bz)
#         #
#         # data_parse.get_model([a, b, c])
#         # print(lines, a, b, c)
#
