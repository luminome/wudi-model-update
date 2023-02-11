#!/usr/bin/env python3
import numpy as np
from scipy.spatial import Delaunay


def bokeh_plot_simple(points_dicts: list, lines_dicts: list, plot_label=None):
    from bokeh.plotting import figure, output_file, show, save
    from bokeh.models import ColumnDataSource
    from bokeh.models.tools import HoverTool

    from bokeh.io import reset_output
    output_file("bokeh_plot_points.html")

    p = figure(title=str(plot_label), match_aspect=True, active_scroll="wheel_zoom", lod_threshold=None)

    for points_set in points_dicts:
        points_set['source'] = ColumnDataSource(data=points_set['data'])
        # add a circle renderer with a size, color, and alpha
        points_set['plot'] = p.circle(
            x='x_values',
            y='y_values',
            size='size',
            source=points_set['source'],
            color='color_values',
            line_width=0,
            alpha=0.5)

        p.add_tools(HoverTool(
            renderers=[points_set['plot']],
            tooltips=[('z', '@z_values'), ('p', points_set['name'])]))

    for lines_set in lines_dicts:
        line_id, line_x, line_y, length, line_color, coords = [], [], [], [], [], []
        t_color = 'blue'
        for i, line in enumerate(lines_set['lines']):
            line_id.append(i)
            line_color.append(t_color)
            line_x.append(list(line.coords.xy[0]))
            line_y.append(list(line.coords.xy[1]))
            length.append(line.length)
            coords.append(len(line.coords))
        lines_set['source'] = ColumnDataSource(dict(x=line_x, y=line_y, id=line_id, color=line_color, length=length, coords=coords))
        lines_set['plot'] = p.multi_line(
            'x',
            'y',
            source=lines_set['source'],
            line_color="color",
            line_width=1.0
        )

        p.add_tools(HoverTool(
            renderers=[lines_set['plot']],
            tooltips=[('id', '@id'), ('length', '@length'), ('coords', '@coords'), ('p', lines_set['name'])]))

    save(p)
    reset_output()


# https://stackoverflow.com/questions/41200719/how-to-get-all-array-edges
def border_elems(a, W):  # Input array : a, Edgewidth : W
    n = a.shape[0]
    r = np.minimum(np.arange(n)[::-1], np.arange(n))
    return a[np.minimum(r[:, None], r) < W]


def get_depth_points(data, sector, level) -> np.ndarray:
    import config as conf
    from scipy.interpolate import interpn

    degs = conf.master_degree_intervals[0]
    levels_mod = [15, 10, 6, 2, 1]

    r = 60
    q = levels_mod[level] / degs

    data_origin = [-6, 46, 36.5, 30]

    sample_lon = (sector.loc[0]-data_origin[0])
    sample_lat = (data_origin[1]-sector.loc[1])

    #print('relative', sample_lon, sample_lat)

    arm = data['data']
    x_range = np.arange(0, arm.shape[1], 1)
    y_range = np.arange(0, arm.shape[0], 1)

    lon_start = [sample_lon*r, (sample_lon+degs) * r]
    lat_start = [sample_lat*r, (sample_lat+degs) * r]
    #print(lon_start, lat_start)

    k_o = r/60  #1/2  #0  # 1/60
    x_pts = np.linspace(lon_start[0] - k_o, lon_start[1] - k_o, num=int(r/q)+1, endpoint=True)
    y_pts = np.linspace(lat_start[0] - k_o, lat_start[1] - k_o, num=int(r/q)+1, endpoint=True)

    points_x, points_y = np.meshgrid(x_pts, y_pts)
    #print('grid', points_x.shape)

    k = points_x.size
    d = [points_x.reshape(1, k)[0], points_y.reshape(1, k)[0]]
    check_points = list(zip(d[1], d[0]))

    k_in = interpn([y_range, x_range], arm, check_points, method='linear', bounds_error=False, fill_value=None)
    #print(k_in.shape)
    #print('result min, max', k_in[0], k_in[-1])

    ost = [-6, 46]
    st_lon = np.linspace(ost[0]+(lon_start[0]/r), ost[0]+(lon_start[1]/r), num=int(r/q)+1, endpoint=True)
    st_lat = np.linspace(ost[1]-(lat_start[0]/r), ost[1]-(lat_start[1]/r), num=int(r/q)+1, endpoint=True)
    st_x, st_y = np.meshgrid(st_lon, st_lat)

    k = st_x.size
    d = [st_x.reshape(1, k)[0], st_y.reshape(1, k)[0], k_in]

    pck = np.column_stack((d[0], d[1], d[2]))
    # pck = np.delete(pck, np.where((pck[:, 0] < -6) | (pck[:, 1] > 46))[0], axis=0)

    kl = points_x.shape[0]
    index = np.arange(0, kl*kl, 1).reshape(kl, kl)
    filter_indices = border_elems(index, 1)

    d_pck = pck[filter_indices]
    d_pck = np.delete(d_pck, np.where((d_pck[:, 0] < -6) | (d_pck[:, 1] > 46))[0], axis=0)
    d_pck = np.delete(d_pck, (np.where(d_pck[:, 2] > 0)[0]), axis=0)

    return d_pck


def raw_delaunay(xa, ya, za):
    x = np.array(xa)
    y = np.array(ya)
    z = np.array(za)
    xy = np.stack((x, y), axis=1)

    tri = Delaunay(xy, furthest_site=False, qhull_options="Qi")
    mask = np.ones((tri.simplices.shape[0]), dtype=bool)

    for n, s in enumerate(tri.simplices):
        mask[n] = (z[s[0]] == 0.0 and z[s[1]] == 0.0 and z[s[2]] == 0.0)

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_trisurf(x, y, z, triangles=tri.simplices, mask=mask, cmap=plt.cm.Spectral)
    # print(tri.simplices.shape)
    # print(tri.simplices[np.logical_not(mask)].shape)
    # plt.show()

    return tri.simplices[np.logical_not(mask)]


def test_delaunay(xa, ya, za):
    import matplotlib.pyplot as plt
    x = np.array(xa)
    y = np.array(ya)
    z = np.array(za)

    ax = x
    ay = y
    az = z

    xy = np.stack((x, y), axis=1)
    tri = Delaunay(xy, furthest_site=False, qhull_options="Q3 Q4 Q6 Q7 Q8 Qi QJ Pp")  #//, qhull_options="QJ Pp"

    k_tri = []
    #
    # print(z.tolist())
    mask = []

    for s in tri.simplices:
        kt = z[s[0]]
        if z[s[1]] == kt and z[s[2]] == kt:
            mask.append(True)
        else:
            mask.append(False)

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot_trisurf(x, y, z, triangles=tri.simplices, mask=mask, cmap=plt.cm.Spectral)

    #
    # print(tri.simplices.shape)
    print(tri.simplices[np.logical_not(mask)])

    plt.show()

    return tri.simplices[np.logical_not(mask)]


def get_sector_depth_map(depth_source, contours_at_level, sector, level) -> dict:
    from shapely.geometry import LineString
    import utilities as util
    import config as conf
    #contours_at_level = contours_source[int(level)]

    def build_vertex_collection(depth: float, group: list, collector, record):
        for c in group:
            f = util.poly_s_to_list(c)
            for n, raw_line in enumerate(f):
                #line = line.simplify(0.005)  #oversimplify gets gaps in mesh
                distance_delta = conf.contour_ranges['line_density']
                if raw_line.length >= distance_delta:

                    distances = np.arange(0, raw_line.length, distance_delta)
                    points = [raw_line.interpolate(distance) for distance in distances]

                    if len(raw_line.boundary.geoms):
                        points += [raw_line.boundary.geoms[1]]
                    else:
                        points += [raw_line.coords[0]]

                    if len(points) > 4:
                        line = LineString(points)

                        start_index = len(collector['x'])
                        for co in line.coords:
                            collector['x'].append(co[0])
                            collector['y'].append(co[1])
                            collector['z'].append(depth)
                        end_index = len(collector['x'])

                        line_stat = {
                            "depth": depth,
                            "len": util.value_cleaner(line.length, 5),
                            "coords": len(line.coords),
                            "start": start_index,
                            "end": end_index
                        }

                        collector['lines'].append(line)
                        record.append(line_stat)
                        #print(line_stat)

    points_collector = {"x": [], "y": [], "z": [], "lines": []}
    lines_record = []
    labels_collection = []

    for depth_level, contour_depth in enumerate(contours_at_level):
        contour_labels_indices = [r for (r, k) in enumerate(contour_depth['labels']) if k.intersects(sector.box)]
        contour_labels = [sector.box.intersection(contour_depth['labels'][p]) for p in contour_labels_indices]
        if len(contour_labels):
            labels_collection.append({
                'd': contour_depth['d'],
                'labels': [util.geometry_to_coords(fp) for fp in contour_labels]})

        contour_indices = [r for (r, k) in enumerate(contour_depth['contours']) if k.intersects(sector.box)]
        if len(contour_indices):
            #print(f"Depth-level: {depth_level} ({contour_depth['d']}m)")
            contours = [sector.box.intersection(contour_depth['contours'][p]) for p in contour_indices]
            build_vertex_collection(contour_depth['d'], contours, points_collector, lines_record)

    if len(lines_record) == 0:
        return {"contour_lines": None}

    grid_array_d = get_depth_points(depth_source, sector, int(level))

    contour_vt = np.stack((points_collector['x'], points_collector['y']), axis=1)

    for r in grid_array_d:
        points_collector['x'].append(r[0])
        points_collector['y'].append(r[1])
        points_collector['z'].append(r[2])

    indices = raw_delaunay(points_collector['x'], points_collector['y'], points_collector['z'])
    contour_vt_xy = contour_vt.flatten()
    depth_vt_xyz = grid_array_d.flatten()

    packet = {
        "contour_lines": lines_record,
        "verts_xy": contour_vt_xy.round(4).tolist(),
        "verts_xyz": depth_vt_xyz.tolist(),
        "indices": indices.flatten().tolist(),
        "labels": labels_collection
    }

    return packet


def get_sector_depth_map_alt(depth_source, contours_at_level, sector, level, limit: tuple) -> tuple:
    from shapely.geometry import LineString
    import utilities as util
    import config as conf
    #contours_at_level = contours_source[int(level)]

    def build_vertex_collection(depth: float, group: list, collector, record):
        for c in group:
            f = util.poly_s_to_list(c)
            for n, raw_line in enumerate(f):
                #line = line.simplify(0.005)  #oversimplify gets gaps in mesh
                distance_delta = conf.contour_ranges['line_density']
                if raw_line.length >= distance_delta:

                    distances = np.arange(0, raw_line.length, distance_delta)
                    points = [raw_line.interpolate(distance) for distance in distances]

                    if len(raw_line.boundary.geoms):
                        points += [raw_line.boundary.geoms[1]]
                    else:
                        points += [raw_line.coords[0]]

                    if len(points) > 4:
                        line = LineString(points)

                        start_index = len(collector['x'])
                        for co in line.coords:
                            collector['x'].append(co[0])
                            collector['y'].append(co[1])
                            collector['z'].append(depth)
                        end_index = len(collector['x'])

                        line_stat = {
                            "depth": depth,
                            "len": util.value_cleaner(line.length, 5),
                            "coords": len(line.coords),
                            "start": start_index,
                            "end": end_index
                        }

                        collector['lines'].append(line)
                        record.append(line_stat)
                        #print(line_stat)

    points_collector = {"x": [], "y": [], "z": [], "lines": []}
    lines_record = []
    labels_collection = []

    for depth_level, contour_depth in enumerate(contours_at_level):
        # contour_labels_indices = [r for (r, k) in enumerate(contour_depth['labels']) if k.intersects(sector.box)]
        # contour_labels = [sector.box.intersection(contour_depth['labels'][p]) for p in contour_labels_indices]
        # if len(contour_labels):
        #     labels_collection.append({
        #         'd': contour_depth['d'],
        #         'labels': [util.geometry_to_coords(fp) for fp in contour_labels]})

        if limit[0] <= depth_level <= limit[1]:
            contour_indices = [r for (r, k) in enumerate(contour_depth['contours']) if k.intersects(sector.box)]
            if len(contour_indices):
                print(f"Depth-level: {depth_level} ({contour_depth['d']}m)")
                contours = [sector.box.intersection(contour_depth['contours'][p]) for p in contour_indices]
                build_vertex_collection(contour_depth['d'], contours, points_collector, lines_record)

    if len(lines_record) == 0:
        return {"contour_lines": None}
    #
    # grid_array_d = get_depth_points(depth_source, sector, int(level))
    #
    # contour_vt = np.stack((points_collector['x'], points_collector['y']), axis=1)
    #
    # for r in grid_array_d:
    #     points_collector['x'].append(r[0])
    #     points_collector['y'].append(r[1])
    #     points_collector['z'].append(r[2])
    #
    # indices = raw_delaunay(points_collector['x'], points_collector['y'], points_collector['z'])
    # contour_vt_xy = contour_vt.flatten()
    # depth_vt_xyz = grid_array_d.flatten()
    #
    # packet = {
    #     "contour_lines": lines_record,
    #     "verts_xy": contour_vt_xy.round(4).tolist(),
    #     "verts_xyz": depth_vt_xyz.tolist(),
    #     "indices": indices.flatten().tolist(),
    #     "labels": labels_collection
    # }

    return points_collector, lines_record


def test_depth(lon: str, lat: str, level: str = 0):
    # import numpy as np
    import config as conf
    from builder import sectorize
    from builder import load
    import utilities as util

    def ost(d):
        if d % conf.master_degree_intervals[0] != 0:
            d -= d % conf.master_degree_intervals[0]
        return d

    width = np.ceil(conf.master_bounds[1] * (1 / conf.master_degree_intervals[0]))
    dx = int(lon) - conf.master_bounds[0][0]
    dy = conf.master_bounds[0][3] - int(lat)
    s = int((width * ost(dy) + ost(dx)) * (1 / conf.master_degree_intervals[0]))

    test_sector = sectorize.init_sector(s, width, conf.master_degree_intervals[0], conf.master_bounds[0])
    print(f"Sector Nº{test_sector.s_id} Loc:{test_sector.loc} Level:{level}")

    depth_source = load.depth_dict()
    print(f"depth source data: {depth_source['data'].shape}")

    depth_contour_levels = load.depth_contours()

    contours_at_level = depth_contour_levels[int(level)]

    points_collector, lines_record = get_sector_depth_map_alt(depth_source, contours_at_level, test_sector, int(level), (0, 4,))
    print(lines_record)
    # print(pck)

    ad_z = [3.0 for v in points_collector['z']]
    data = {'x_values': points_collector['x'],
            'y_values': points_collector['y'],
            'z_values': points_collector['z'],
            'color_values': ['red'] * len(points_collector['z']),
            'size': ad_z}

    points_one = {"name": "contour-vertex", "data": data}
    lines_one = {"name": "contour", "lines": points_collector['lines']}

    # grid_array_d = get_depth_points(depth_source, test_sector, int(level))
    #
    # a = grid_array_d
    #
    # ad = [a[:, 0], a[:, 1], a[:, 2]]
    #
    # z_max = np.max(ad[2])
    # z_min = np.min(ad[2])
    #
    # ad_z = [util.normalize_val(v, z_min, z_max)*10 for v in ad[2]]
    #
    # data = {'x_values': ad[0],
    #         'y_values': ad[1],
    #         'z_values': ad[2],
    #         'color_values': ['green'] * len(ad[0]),
    #         'size': ad_z}
    #
    # points_two = {"name": "depth-edges", "data": data}

    indices = test_delaunay(points_collector['x'], points_collector['y'], points_collector['z'])

    bokeh_plot_simple([points_one], [lines_one], 'test scenario')
    #
    # contour_verts = np.stack((points_collector['x'], points_collector['y']), axis=1)
    #
    # for r in grid_array_d:
    #     points_collector['x'].append(r[0])
    #     points_collector['y'].append(r[1])
    #     points_collector['z'].append(r[2])
    #
    # indices = raw_delaunay(points_collector['x'], points_collector['y'], points_collector['z'])
    #
    # print(indices.flatten())
    #
    # print(contour_verts.shape)
    # print(grid_array_d.shape)
    # print(len(points_collector['x']))
    #
    # contour_verts_xy = contour_verts.flatten()
    # depth_verts_xyz = grid_array_d.flatten()
    #
    # packet = {
    #     "contour_lines": lines_record,
    #     "verts_xy": contour_verts_xy.round(4).tolist(),
    #     "verts_xyz": depth_verts_xyz.tolist()
    # }
    #
    # print(packet)
    #
    # exit()


    pass
