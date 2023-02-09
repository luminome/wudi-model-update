#!/usr/bin/env python3

import os
import config as conf
import utilities as util
from typing import List
from builder import load
from builder import place_data_acquire


def map_geometry() -> List:
    import geopandas as gpd
    from shapely.geometry import box

    min_x, min_y, max_x, max_y = conf.map_bounds_degrees
    geographic_bounds = box(min_x, min_y, max_x, max_y)

    geo_bounds = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geographic_bounds])
    map_regions = gpd.read_file(os.path.join(conf.data_resource_path, 'GOaS_v1_20211214/goas_v01.shp'))
    map_med = gpd.overlay(geo_bounds, map_regions, how='intersection')

    group = map_med['geometry']
    map_multi_poly = geographic_bounds

    for g in group:
        print(type(g))
        for poly in util.poly_s_to_list(g):
            map_multi_poly = map_multi_poly.difference(poly)

    return [util.save_asset(map_multi_poly, 'map-geometry')]


def eco_regions() -> List:
    import geopandas as gpd
    regions = gpd.read_file(os.path.join(conf.data_resource_path, 'MEOW/meow_ecos.shp'))
    return [util.save_asset(regions[regions.PROVINCE == "Mediterranean Sea"], 'map-eco-regions')]


def geo_names() -> List:
    import numpy as np
    import pandas as pd

    geonames = np.genfromtxt(os.path.join(conf.source_path, 'GEO_OBJ_NAME_v2-2.csv'), dtype='str', encoding='UTF-8')
    index_values = np.arange(1, geonames.size+1, 1, dtype=int)
    column_values = ['geoname']
    master_geonames = pd.DataFrame(
        data=geonames,
        index=index_values,
        columns=column_values)

    return [util.save_asset(master_geonames, 'map-geo-names')]


def geo_associations() -> List:
    import pandas as pd
    from shapely.geometry import Point

    wudi_df = pd.read_pickle(os.path.join(conf.support_path, conf.support_files['wudi-points']))
    places_df = load.places()
    protected_df = load.protected_areas()

    all_wudi_points = []
    for n in range(wudi_df.shape[0]):
        point = Point(wudi_df.iloc[n].M_lon, wudi_df.iloc[n].M_lat)
        all_wudi_points.append(point)

    r_meta_wudi = {}
    for w in range(wudi_df.shape[0]):
        r_meta_wudi[w] = {
            'pid': int(wudi_df.iloc[w].pid),
            'unique_protected': None,
            'nearest_protected': None,
            'within_protected': None,
            'unique_place': None,
            'nearest_place': None,
            'country': None
        }

    def get_closest_places():
        closest_places = []
        cols = ['lon', 'lat', 'population', 'place_id', 'countryLabel']

        for k, p in enumerate(all_wudi_points):
            s_d = 0.25
            dist = []
            rf = places_df[[cols[0], cols[1], cols[2], cols[3], cols[4]]][((places_df[cols[0]] > p.x - s_d) & (places_df[cols[0]] < p.x + s_d)) & (
                    (places_df[cols[1]] > p.y - s_d) & (places_df[cols[1]] < p.y + s_d))]

            for wi, row in rf.iterrows():
                dist.append([k, int(row.place_id), Point(row.lon, row.lat).distance(p), int(row.population), row.countryLabel])

            by_distance = sorted(dist, key=lambda x: (x[3], x[2]))[::-1]

            if len(by_distance):  # and by_distance[0][2] < 0.125:
                closest_places.append(by_distance[0])  #for all of these closest points

            util.show_progress(f"places", k, len(all_wudi_points))

        for r in closest_places:
            r_meta_wudi[r[0]]['nearest_place'] = r[1]
            r_meta_wudi[r[0]]['country'] = r[4]

        #eliminate redundancy here
        d_filter = {}
        for r in closest_places:
            if not r[1] in d_filter:
                d_filter[r[1]] = []
            d_filter[r[1]].append(r)

        for rf in d_filter.keys():
            reso = sorted(d_filter[rf], key=lambda x: x[2])
            if len(reso):
                r_meta_wudi[reso[0][0]]['unique_place'] = reso[0][1]

    def get_closest_regions():
        cols = {'lon': [], 'lat': []}

        for wi, row in protected_df.iterrows():
            cen = row['CENTROID'].split(',')
            cols['lon'].append(float(cen[0]))
            cols['lat'].append(float(cen[1]))

        protected_df['lon'] = cols['lon']
        protected_df['lat'] = cols['lat']

        cols = ['lon', 'lat']
        filtered = {}

        for k, p in enumerate(all_wudi_points):
            s_d = 1.0
            dist = []
            rf = protected_df[((protected_df[cols[0]] > p.x - s_d) & (protected_df[cols[0]] < p.x + s_d)) & (
                    (protected_df[cols[1]] > p.y - s_d) & (protected_df[cols[1]] < p.y + s_d))]

            contained = []

            closest = None

            for wi, row in rf.iterrows():
                dist.append([k, wi, Point(row.lon, row.lat).distance(p)])
                if row.geometry.contains(p):
                    contained.append(wi)
                res = sorted(dist, key=lambda x: x[2])
                closest = res[0]

                if wi not in filtered:
                    filtered[wi] = []
                if closest[1] == wi:
                    filtered[wi].append(closest)

            if closest is not None:
                r_meta_wudi[k]['nearest_protected'] = closest[1]
            if len(contained) > 0:
                r_meta_wudi[k]['within_protected'] = contained

            util.show_progress(f"protected", k, len(all_wudi_points))
            # print(k, closest, contained)

        for k in filtered.keys():
            res = sorted(filtered[k], key=lambda x: x[2])
            if len(res):
                r_meta_wudi[res[0][0]]['unique_protected'] = k

    get_closest_places()

    get_closest_regions()

    for k in r_meta_wudi.keys():
        print(k, r_meta_wudi[k])

    tabled = pd.DataFrame.from_dict(r_meta_wudi)
    tabled_transposed = tabled.transpose()
    return [util.save_asset(tabled_transposed, 'map-geo-associations')]


def depth_dict() -> List:
    import numpy as np
    parts = {
        "data": 'bathy_med_tt.csv',
        "lons": 'bathy_lon_vector_t.csv',
        "lats": 'bathy_lat_vector_t.csv'
    }

    depth_points = {
        "reso": [10, 6, 2, 1],
        "degs": 60,
        "levels": 4,
        "dimension": 1
    }

    for k in parts.keys():
        depth_points[k] = np.loadtxt(
            open(os.path.join(conf.data_resource_path, parts[k]), "rb"),
            delimiter=";",
            encoding=None,
            skiprows=0).tolist()

    depth_points['origin'] = util.get_data_scale_extents(depth_points)

    return [util.save_asset(depth_points, 'map-depth-points')]


def depth_contours() -> List:
    import numpy as np
    from scipy.ndimage.filters import gaussian_filter
    from shapely.affinity import translate, scale
    from shapely.geometry import box, LineString
    from skimage import measure

    depth_source = load.depth_dict()
    eco_regions_mask = load.eco_regions()
    points_data = depth_source['data']
    poly_origin = depth_source['origin']
    contours_all = [[]] * conf.levels_range
    iso_bath = []

    def contour_labels_getter(contours):
        contours_labels = []
        for c in contours:
            kpt = util.poly_s_to_list(c)

            for li in kpt:
                o = conf.contour_ranges['label_density']
                d = conf.contour_ranges['label_size']
                a = np.ceil(li.length * o)
                m = li.length * o
                stills = np.arange(0, m, m / a)

                if li.length > conf.contour_ranges['label_min_length']:
                    for s in stills:
                        a = li.interpolate((s / o) - d)
                        b = li.interpolate((s / o))
                        c = li.interpolate((s / o) + d)
                        contours_labels.append(LineString([a, b, c]))

                    # //contours_labels.append(labels_points)
                    # contours_labels.append(MultiLineString(labels_points))

        return contours_labels

    def contour_getter(data, level):
        f_contours = measure.find_contours(data, level)
        contours = []
        for ep in f_contours:
            ep = LineString(np.flip(ep))
            ep = scale(ep, xfact=1 / 60, yfact=-1 / 60, origin=(0, 0))
            ep = translate(ep, xoff=poly_origin[0], yoff=poly_origin[1])
            ep = ep.intersection(eco_regions_mask)
            contours.append(ep)
        return contours

    for ra in range(conf.levels_range):
        contours_all[ra] = []
        g_data = gaussian_filter(points_data, sigma=conf.contour_ranges["filter"][ra])
        g_range = np.arange(0, conf.contour_ranges["depth_max"], conf.contour_ranges["depth_interval"][ra])

        start = []  #g_range  #[conf.contour_ranges["depth_interval"][ra]]

        for r in g_range:
            start.append(-r)
        #
        # print(start)

        for i, lv in enumerate(start):
            clutch = contour_getter(g_data, lv)
            labels = contour_labels_getter(clutch)
            contours_all[ra].append({'d': float(lv), 'contours': clutch, 'labels': labels})
            util.show_progress(f"generate contours {ra} {lv}m", i, len(start))

    dep = conf.contour_ranges["iso_bath_depth"]
    print('generating iso_bath', dep)
    g_data = gaussian_filter(points_data, sigma=conf.contour_ranges["iso_bath_filter"])
    iso_bath.append({'d': dep, 'contours': contour_getter(g_data, dep)})

    return [
        util.save_asset(contours_all, 'map-depth-contours'),
        util.save_asset(iso_bath, f'map-iso-bath-{abs(dep)}')
    ]


def places(has_overpass: bool = False) -> List:
    import pandas as pd
    import json

    df = pd.read_pickle(os.path.join(conf.support_path, 'wudi-points-Dataframe.pkl'))
    result_one = 'map-places-overpass loaded.'
    if not has_overpass:
        places_overpass = place_data_acquire.get_overpass_data(df)
        result_one = util.save_asset(places_overpass, 'map-places-overpass')

    src = os.path.join(conf.support_path, 'map-places-overpass-dict.json')
    with open(src) as depth_file:
        places_overpass = json.load(depth_file)

    places_overpass_wiki = place_data_acquire.get_wiki_media_data(places_overpass)
    result_two = util.save_asset(places_overpass_wiki, 'map-places-overpass-wiki')

    return [result_one, result_two]


def protected_areas() -> List:
    import csv
    import fiona
    import geopandas as gpd
    import pandas as pd
    import numpy as np

    area_filter = [
        "geometry",
        "NAME",
        "STATUS_YR",
        "STATUS_ENG",
        "MAPAMED_ID",
        "PARENT_ID",
        "REP_AREA",
        "SITE_TYPE_ENG",
        "DESIG_ENG",
        "IUCN_CAT_ENG",
        "WEBSITE"
    ]

    criteria_filter = ['Marine N2000 Proposed Site', 'MPA of National Statute']  #// WEIRD

    def load_sources():
        country_file = os.path.join(conf.data_resource_path, 'MAPAMED_2019_edition/mapamed_2019_dataset_definition_countries.tsv')
        with open(country_file) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            country_data = [row for row in reader]

        shapes_file = os.path.join(conf.data_resource_path, 'MAPAMED_2019_edition/MAPAMED_2019_spatial_data_epsg3035.gpkg')
        gpkg_tables = {}

        for layer_name in fiona.listlayers(shapes_file):
            print(layer_name)
            g_layer = gpd.read_file(shapes_file, layer=layer_name)
            print('converting crs to 4326')
            g_layer = g_layer.to_crs(crs=4326)
            gpkg_tables[layer_name] = g_layer

        return gpkg_tables, country_data

    def regions_parse(main_tables, country_data):
        regions_all = []
        elements_all = []
        for k, v in main_tables.items():

            print('loading regions', k)
            if k == 'Scope of the Barcelona Convention (IHO-MSFD)':
                regions_all = v.to_dict('records')

            print('loading elements', k)
            if k == 'MAPAMED 2019 edition - EPSG:3035':
                indx = v.shape[0]
                v = v.fillna(np.nan).replace([np.nan], [None])
                # HIFIVE TO SELF ^
                mct = [v.iloc[i] for i in range(0, indx) if v.iloc[i]['DESIG_CAT_ENG'] in criteria_filter]
                max_area = 0
                avg_area = 0
                for mv in mct:
                    compare = mv['REP_AREA'] if mv['REP_AREA'] is not None else 0.0
                    avg_area += compare

                    if compare > max_area:
                        max_area = mv['REP_AREA']

                avg_area /= len(mct)

                print(avg_area, 'km')
                print(max_area, 'km')

                for i, elem in enumerate(mct):
                    area = {}
                    for f in area_filter:
                        area[f] = elem[f]

                    if type(area['REP_AREA']) == float:
                        nsca = util.normalize_val(area['REP_AREA'], 0.1, avg_area)
                        area['scale'] = 1 if np.isnan(nsca) or nsca < 0 else np.ceil(nsca) if nsca < 4 else 4
                    else:
                        area['scale'] = 1

                    area['CENTROID'] = np.array(elem['geometry'].centroid.coords[0])
                    area['LON'] = np.array(elem['geometry'].centroid.x)
                    area['LAT'] = np.array(elem['geometry'].centroid.y)

                    area['COUNTRY'] = [
                        p['COUNTRY_ENG'] for p in country_data if p['ISO3'] in elem['ISO3'][1:-1]
                    ]
                    area['MED_REGION'] = [
                        p['NAME_ENG'] for p in regions_all if p['MSFD_REGION'] in elem['MSFD_REGION'][1:-1]
                    ]

                    elements_all.append(area)

        return {'protected': elements_all, 'eco_regions_mapamed': regions_all}

    tables, country_info = load_sources()
    ref_table = regions_parse(tables, country_info)
    protected_df = pd.json_normalize(ref_table['protected'])
    map_a_med_regions_df = pd.json_normalize(ref_table['eco_regions_mapamed'])

    return [
        util.save_asset(protected_df, 'map-protected-areas-raw'),
        util.save_asset(map_a_med_regions_df, 'map-protected-areas-regions')
    ]


def map_layers() -> List:
    from shapely.geometry import LineString

    depth_contour_levels = load.depth_contours()
    eco_regions_mask = load.eco_regions()
    map_levels = load.map_geometry()

    map_sets = []

    for lv, map_level in enumerate(map_levels):
        p_polys = [p for p in map_level.geoms]
        p_lines = [LineString(ref.exterior.coords).intersection(eco_regions_mask) for ref in map_level.geoms]
        map_sets.append({'polygons': p_polys, 'line_strings': p_lines, 'contours': depth_contour_levels[lv]})
        print('level', lv, len(p_polys), len(p_lines))

    return [util.save_asset(map_sets, 'map-sector-layers')]


def tests():
    # print(geo_associations())
    # print(places(True))
    print(depth_contours())
    # print(map_geometry())
    # print(map_layers())
    # ape = load.depth_contours()
    # print("ape[0][0]['labels'][0].geoms[0]")
    # print(len(ape), ape[0][0]['labels'][0].geoms[0])
    # print(depth_dict())
    # print(depth_contours())
    print('not testing')
    pass

