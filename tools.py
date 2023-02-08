#!/usr/bin/env python3
import numpy as np


def bokeh_plot_points(points_data_dict: dict, plot_label=None):
    from bokeh.plotting import figure, output_file, show
    from bokeh.models import GeoJSONDataSource, ColumnDataSource
    from bokeh.models.tools import HoverTool, WheelZoomTool, PanTool, CrosshairTool, LassoSelectTool

    p = figure(title=str(plot_label), match_aspect=True, active_scroll="wheel_zoom", lod_threshold=None)

    source = ColumnDataSource(data=points_data_dict)
    # add a circle renderer with a size, color, and alpha
    circles = p.circle(x='x_values', y='y_values', size='size', source=source, color='red', line_width=0, alpha=0.5)

    p.add_tools(HoverTool(renderers=[circles], tooltips=[('z', '@z_values')]))

    show(p)


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

    k = st_x.size
    d = [st_x.reshape(1, k)[0], st_y.reshape(1, k)[0], k_in]

    pck = np.column_stack((d[0], d[1], d[2]))
    print(pck.shape)

    #pck = np.delete(pck, np.where((pck[:, 0] >= 25) & (pck[:, 0] <= 35))[0], axis=0)
    pck = np.delete(pck, np.where((pck[:, 0] < -6) | (pck[:, 1] > 46))[0], axis=0)

    # pck = np.delete(pck, (np.where(pck < -6)[0]), axis=0)
    # print(pck.shape)
    # print(pck[0])
    #
    # d = [pck[:, 0], pck[:, 1], pck[:, 2]]

    return pck


def test_depth(lon: str, lat: str, level: str = 0):
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
    print(test_sector.loc, test_sector.s_id, test_sector.box)

    depth_source = load.depth_dict()
    print(depth_source['data'].shape)

    depth_contour_levels = load.depth_contours()

    contours_at_level = depth_contour_levels[int(level)]

    for depth, contour_depth in enumerate(contours_at_level):
        print(depth, contour_depth['d'])
        contour_indices = [r for (r, k) in enumerate(contour_depth['contours']) if k.intersects(test_sector.box)]
        contours = [test_sector.box.intersection(contour_depth['contours'][p]) for p in contour_indices]

        #//convert these objects to coordinate columns.
        #contour_coords = [util.geometry_to_coords(fp) for fp in contours]

        print(contours)

    exit()

    array_d = get_depth_points(depth_source, test_sector, int(level))
    ad = [array_d[:, 0], array_d[:, 1], array_d[:, 2]]

    z_max = np.max(ad[2])
    z_min = np.min(ad[2])

    ad_z = [util.normalize_val(v, z_min, z_max)*10 for v in ad[2]]

    data = {'x_values': ad[0],
            'y_values': ad[1],
            'z_values': ad[2],
            'size': ad_z}

    bokeh_plot_points(data, 'test scenario')

    #//isolate depth points interpolated.
    #//isolate depth contours.




    pass
