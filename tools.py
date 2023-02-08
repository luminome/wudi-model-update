#!/usr/bin/env python3

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


def test_depth(lon: str, lat: str, depth: str = 0):
    import numpy as np
    import config as conf
    from builder import sectorize

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

    #//isolate depth contours.
    #//isolate depth points interpolated.



    pass
