source_path = './data/source'
support_path = './support'
data_resource_path = './data/data-resources'
static_data_path = './data/static-build-products'

wudi_UPWthr = 0.4325
wudi_DNWthr = -0.3905
wudi_event_num_days = 7
wudi_database_path = './data/output-databases/wudi.db'
wudi_model_database_path = './data/output-databases/map.db'
wudi_points_masked_by_eco_region = True

sector_save_layers = ['map_polygons', 'map_lines', 'depth_contours', 'depth_maps', 'protected_areas']

db_float_precision = 4

db_tables = {
    'wudi_daily': """
            create table IF NOT EXISTS wudi_daily
            (
                tim INTEGER,
                pid INTEGER,
                raw REAL,
                evt INTEGER
            )
        """,
    #return_object.append([int(time_record), n, pos_sum[n], neg_sum[n], [up_max[n], up_mean[n], down_max[n], down_mean[n]], t_evt])
    'wudi_derivative': """
            create table IF NOT EXISTS wudi_derivative
            (
                tim INTEGER,
                pid INTEGER,
                u_tl INTEGER,
                d_tl INTEGER,
                raw BLOB,
                e_ct INTEGER,
                e_ls BLOB
            )
        """,
    #return_object_meta.append([int(time_record), long, up_max, up_mean, down_max, down_mean])
    'wudi_derivative_meta': """
        create table IF NOT EXISTS wudi_derivative_meta
        (
            tim INTEGER,
            siz INTEGER,
            u_mx INTEGER,
            u_me INTEGER,
            d_mx INTEGER,
            d_me INTEGER
        )
    """,
}

master_bounds = [[-7.0, 29.0, 37.0, 49.0], 44.0, 20.0]

master_degree_intervals = [2]  #, 1, 0.5]

map_bounds_degrees = [-12, 26, 44, 50]


area_limits = [
    0.01,
    0.005
]

simp_limits = [
    0.1,
    0.0001
]

mpa_simp_limits = [
    0.01,
    0.0001
]

pop_limits = [
    5000,
    50000,
    500000,
]

min_population = 3000

levels_range = 5

contour_ranges = {
    "filter": [
        2.5,
        1.5,
        0.5,
        0.1,
        0.05
    ],
    "iso_bath_filter": 0.1,
    "iso_bath_depth": 200.0,
    "depth_interval": [
        1600,
        800,
        400,
        200,
        100
    ],
    "depth_max": 5400,
    "label_min_length": 0.5,
    "label_density": 0.5,
    "label_size": 0.015,
    "line_density": 0.015,  #0.03625
    # "depth_max": 5000
}


support_files = {
    "map-depth-contours": "map-depth-contours-list.pkl",
    "map-depth-points": "map-depth-points-dict.json",
    "map-eco-regions": "map-eco-regions-GeoDataFrame.pkl",
    "map-geo-names": "map-geo-names-DataFrame.pkl",
    "map-geometry": "map-geometry-MultiPolygon.pkl",
    "map-iso-bath": f"map-iso-bath-{abs(contour_ranges['iso_bath_depth'])}-list.pkl",
    "map-protected-areas": "map-protected-areas-DataFrame.pkl",
    "map-protected-areas-raw": "map-protected-areas-raw-DataFrame.pkl",
    "map-protected-areas-regions": "map-protected-areas-regions-DataFrame.pkl",
    "map-sector-layers": "map-sector-layers-list.pkl",
    "map-places-overpass-wiki": "map-places-overpass-wiki-list.pkl",
    "map-places-overpass": "map-places-overpass-dict.json",
    "map-places": "map-places-DataFrame.pkl",
    "wudi-events-meta-": "wudi-events-meta-ndarray.npy",
    "wudi-points": "wudi-points-DataFrame.pkl",
    "wudi-points-indices": "wudi-points-indices-list.pkl",
    "wudi-time-index": "wudi-time-index-ndarray.npy",
    "wudi-vectorized": "wudi-vectorized-ndarray.npy",
}


# CREATE INDEX wudi_ids ON wudi_daily (pid);
# CREATE INDEX wudi_times ON wudi_daily (tim);
