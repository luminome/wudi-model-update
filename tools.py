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


def bokeh_plot(lines_iterable: list, points_iterable: list, plot_label=None):
    from bokeh.plotting import figure, output_file, show
    from bokeh.models import GeoJSONDataSource, ColumnDataSource
    from bokeh.models.tools import HoverTool, WheelZoomTool, PanTool, CrosshairTool, LassoSelectTool

    p = figure(title=str(plot_label), match_aspect=True, active_scroll="wheel_zoom", lod_threshold=None)

    line_id, line_x, line_y, length, line_color, coords = [], [], [], [], [], []

    t_color = 'blue'
    plot_data_sources = {}
    for i, line in enumerate(lines_iterable):
        line_id.append(i)
        line_color.append(t_color)
        line_x.append(list(line.coords.xy[0]))
        line_y.append(list(line.coords.xy[1]))
        length.append(line.length)
        coords.append(len(line.coords))

    plot_data_sources['lines'] = ColumnDataSource(dict(x=line_x, y=line_y, id=line_id, color=line_color, length=length, coords=coords))

    point_id, point_x, point_y, point_color = [], [], [], []
    for i, point in enumerate(points_iterable):
        # point_id.append(i)
        # point_color.append(t_color)
        point_x.append(point.x)
        point_y.append(point.y)

    data = {'x_values': point_x,
            'y_values': point_y}

    # create a ColumnDataSource by passing the dict
    source = ColumnDataSource(data=data)

    # add a circle renderer with a size, color, and alpha
    p.circle(x='x_values', y='y_values', size=5.0, source=source, color='red', line_width=0, alpha=0.5)
    # p.circle(x=point_x, y=point_y, size='3', color='red', line_width=0, alpha=0.5)

    line_plot = p.multi_line(
        'x',
        'y',
        source=plot_data_sources['lines'],
        line_color="color",
        line_width=1.0
    )

    p.add_tools(HoverTool(renderers=[line_plot], tooltips=[('id', '@id'), ('length', '@length'), ('coords', '@coords')]))

    show(p)


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

    print('relative', sample_lon, sample_lat)

    arm = data['data']
    x_range = np.arange(0, arm.shape[1], 1)
    y_range = np.arange(0, arm.shape[0], 1)

    lon_start = [sample_lon*r, (sample_lon+degs) * r]
    lat_start = [sample_lat*r, (sample_lat+degs) * r]
    print(lon_start, lat_start)

    k_o = r/60  #1/2  #0  # 1/60
    x_pts = np.linspace(lon_start[0] - k_o, lon_start[1] - k_o, num=int(r/q)+1, endpoint=True)
    y_pts = np.linspace(lat_start[0] - k_o, lat_start[1] - k_o, num=int(r/q)+1, endpoint=True)

    points_x, points_y = np.meshgrid(x_pts, y_pts)
    print('grid', points_x.shape)

    k = points_x.size
    d = [points_x.reshape(1, k)[0], points_y.reshape(1, k)[0]]
    check_points = list(zip(d[1], d[0]))

    k_in = interpn([y_range, x_range], arm, check_points, method='linear', bounds_error=False, fill_value=None)
    print(k_in.shape)
    print('result min, max', k_in[0], k_in[-1])

    ost = [-6, 46]
    st_lon = np.linspace(ost[0]+(lon_start[0]/r), ost[0]+(lon_start[1]/r), num=int(r/q)+1, endpoint=True)
    st_lat = np.linspace(ost[1]-(lat_start[0]/r), ost[1]-(lat_start[1]/r), num=int(r/q)+1, endpoint=True)
    st_x, st_y = np.meshgrid(st_lon, st_lat)

    # prek = k_in.reshape(points_x.shape[0], points_x.shape[1])
    # print(prek[0:1, 0:1])

    k = st_x.size
    d = [st_x.reshape(1, k)[0], st_y.reshape(1, k)[0], k_in]



    pck = np.column_stack((d[0], d[1], d[2]))
    pck = np.delete(pck, np.where((pck[:, 0] < -6) | (pck[:, 1] > 46))[0], axis=0)
    # pck = np.delete(pck, (np.where(pck[:, 2] > 0)[0]), axis=0)

    kl = points_x.shape[0]
    index = np.arange(0, kl*kl, 1).reshape(kl, kl)
    filter_indices = border_elems(index, 1)

    d_pck = pck  #[filter_indices]  # pck.reshape(points_x.shape[0], points_x.shape[1])
    # d_pck = np.delete(d_pck, (np.where(d_pck[:, 2] > 0)[0]), axis=0)
    # print(d_pck)

    # pck = np.delete(pck, (np.where(pck[:, 2] > 0)[0]), axis=0)
    # print(pck.shape)
    # print(pck[0])
    #
    # d = [pck[:, 0], pck[:, 1], pck[:, 2]]

    return pck, d_pck


# https://stackoverflow.com/questions/41200719/how-to-get-all-array-edges
def border_elems(a, W):  # Input array : a, Edgewidth : W
    n = a.shape[0]
    r = np.minimum(np.arange(n)[::-1], np.arange(n))
    return a[np.minimum(r[:, None], r) < W]


def mask_borders(arr, num=1):
    mask = np.zeros(arr.shape, bool)
    for dim in range(arr.ndim):
        mask[tuple(slice(0, num) if idx == dim else slice(None) for idx in range(arr.ndim))] = True
        mask[tuple(slice(-num, None) if idx == dim else slice(None) for idx in range(arr.ndim))] = True
    return mask


#// DO OR DON'T TRUSHT MATPLOTLIB

def test_delaunay(xa, ya, za):
    import matplotlib.pyplot as plt
    x = np.array(xa)
    y = np.array(ya)
    z = np.array(za)

    tri = Delaunay(np.array([x, y]).T, furthest_site=False)

    k_tri = []

    for s in tri.simplices:
        if za[s[0]] <= 0.0 and za[s[1]] <= 0.0 and za[s[2]] <= 0.0:
            k_tri.append(s)
        else:
            print(za[s[0]], za[s[1]], za[s[2]])

    k_tri = np.array(k_tri)

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot_trisurf(x, y, z, cmap=plt.cm.Spectral)

    plt.show()

#
# def plot_delaunay(group: Delaunay, xa, ya, za):
#     import matplotlib.pyplot as plt
#
#     ktri = group.simplices  #[]
#
#     # for s in tri.simplices:
#     #     if z[s[0]] <= 1.0 and z[s[1]] <= 1.0 and z[s[2]] <= 1.0:
#     #         ktri.append(s)
#
#     x = np.array(xa)
#     y = np.array(ya)
#     z = np.array(za)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#
#     # The triangles in parameter space determine which x, y, z points are
#     # connected by an edge
#     #ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)
#     ax.plot_trisurf(x, y, z, triangles=ktri, cmap=plt.cm.Spectral)
#
#     plt.show()



def test_depth(lon: str, lat: str, level: str = 0):

    from shapely.geometry import LineString
    import numpy as np
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

    def build_vertex_collection(depth: float, group: list, collector, record):
        for c in group:
            f = util.poly_s_to_list(c)
            for n, raw_line in enumerate(f):
                #line = line.simplify(0.005)  #oversimplify gets gaps in mesh
                distance_delta = 0.025
                if raw_line.length >= distance_delta:

                    distances = np.arange(0, raw_line.length, distance_delta)
                    points = [raw_line.interpolate(distance) for distance in distances]

                    if len(raw_line.boundary):
                        points += [raw_line.boundary[1]]
                    else:
                        points += [raw_line.coords[0]]

                    if len(points) > 3:
                        line = LineString(points)
                        # line = raw_line

                        start_index = len(collector['x'])
                        for co in line.coords:
                            collector['x'].append(co[0])
                            collector['y'].append(co[1])
                            collector['z'].append(depth)
                            collector['ax'].append(co[0])
                            collector['ay'].append(co[1])
                            collector['az'].append(depth)

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
                        print(line_stat)

    points_collector = {"ax": [], "ay": [], "az": [], "x": [], "y": [], "z": [], "lines": []}
    lines_record = []


    tri = {}

    for depth_level, contour_depth in enumerate(contours_at_level):
        contour_indices = [r for (r, k) in enumerate(contour_depth['contours']) if k.intersects(test_sector.box)]
        if len(contour_indices):
            print(f"Depth-level: {depth_level} ({contour_depth['d']}m)")

            contours = [test_sector.box.intersection(contour_depth['contours'][p]) for p in contour_indices]
            # if depth_level == 0:
            #     build_vertex_collection(1000, contours, points_collector, lines_record)

            build_vertex_collection(contour_depth['d'], contours, points_collector, lines_record)



            # stash = [points_collector["ax"], points_collector["ay"], points_collector["az"]]

    #         if depth_level == 0:
    #             build_vertex_collection(contour_depth['d'], contours, points_collector, lines_record)
    #             tri = Delaunay(np.array([points_collector["ax"], points_collector["ay"], points_collector["az"]]).T, incremental=True)  #, qhull_options="Q12 Q4 QJ Qs")
    #         elif depth_level > 0:
    #             build_vertex_collection(contour_depth['d'], contours, points_collector, lines_record)
    #             tri.add_points(np.array([points_collector["ax"], points_collector["ay"], points_collector["az"]]).T)
    #
    #         points_collector["ax"] = []
    #         points_collector["ay"] = []
    #         points_collector["az"] = []
    #
    #         if depth_level == 3:
    #             pass  #break
    #
    # tri.close()
    # plot_delaunay(tri, points_collector["x"], points_collector["y"], points_collector["z"])
    # exit()

            #
            #     build_vertex_collection(100, contours, points_collector, lines_record)
    # now add bounds

    z_max = np.max(points_collector['z'])
    z_min = np.min(points_collector['z'])
    # ad_z = [util.normalize_val(v, z_min, z_max) * 10 for v in points_collector['z']]
    ad_z = [3.0 for v in points_collector['z']]
    data = {'x_values': points_collector['x'],
            'y_values': points_collector['y'],
            'z_values': points_collector['z'],
            'color_values': ['red'] * len(points_collector['z']),
            'size': ad_z}

    points_one = {"name": "contour-vertex", "data": data}
    lines_one = {"name": "contour", "lines": points_collector['lines']}

    array_d, grid_array_d = get_depth_points(depth_source, test_sector, int(level))

    a = grid_array_d


    ad = [a[:, 0], a[:, 1], a[:, 2]]

    z_max = np.max(ad[2])
    z_min = np.min(ad[2])

    ad_z = [util.normalize_val(v, z_min, z_max)*10 for v in ad[2]]

    data = {'x_values': ad[0],
            'y_values': ad[1],
            'z_values': ad[2],
            'color_values': ['green'] * len(ad[0]),
            'size': ad_z}

    points_two = {"name": "depth-edges", "data": data}

    bokeh_plot_simple([points_one, points_two], [lines_one], 'test scenario')

    for r in a:
        points_collector['x'].append(r[0])
        points_collector['y'].append(r[1])
        points_collector['z'].append(r[2])

    test_delaunay(points_collector['x'], points_collector['y'], points_collector['z'])
    exit()


    #//isolate depth points interpolated.
    #//isolate depth contours.




    pass
