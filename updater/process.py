import os
import config as conf
import utilities as util
from datetime import datetime
import shutil
from updater.db_manage import *

from builder import prepare as prep
from builder import load


def test():
    return 'testing', 'ok'


def wudi_geographical_data_source(new_wudi_netcdf: os.path, force=None) -> dict:
    import geopandas as gpd
    import pandas as pd
    import netCDF4 as nC
    import numpy as np
    from shapely.geometry import Point, box

    #get wudi geographical points data using eco_regions.GeoDataFrame as mask
    def parse_wudi(source_netcdf: os.path, map_regions: gpd.GeoDataFrame) -> pd.DataFrame:
        ds = nC.Dataset(source_netcdf)
        size = np.array(ds['geo']).size
        print('size:', size)
        column_values = ['A_lat', 'A_lon', 'M_lat', 'M_lon', 'B_lat', 'B_lon', 'geo', 'eco', 'pid']
        index_values = np.arange(0, size, 1, dtype=int)
        region_field = np.zeros(size, dtype=int)
        pid_field = np.zeros(size, dtype=int)

        # cols = list(ds.variables)
        # print(list(ds.variables))
        # for n, i in enumerate(cols):
        #     n_arr = np.array(ds[i])
        #     print(n, i, n_arr.shape)

        array = np.array([ds['latlim'][0],
                          ds['lonlim'][0],
                          ds['latitude'],
                          ds['longitude'],
                          ds['latlim'][1],
                          ds['lonlim'][1],
                          ds['geo'],
                          region_field,
                          pid_field],
                         )

        array = np.transpose(array, (1, 0))

        master_dtypes = {
            'A_lat': float,
            'A_lon': float,
            'M_lat': float,
            'M_lon': float,
            'B_lat': float,
            'B_lon': float,
            'geo': int,
            'eco': int,
            'pid': int
        }

        master_wudi = pd.DataFrame(
            data=array,
            index=index_values,
            columns=column_values
        )

        master_wudi = master_wudi.astype(dtype=master_dtypes)

        for wi, row in master_wudi.iterrows():
            check_point = Point(row['M_lon'], row['M_lat'])
            for ri, row_e in map_regions.iterrows():
                test_poly = row_e['geometry']
                if test_poly.contains(check_point):
                    master_wudi.at[wi, 'eco'] = ri
                    master_wudi.at[wi, 'pid'] = wi
                    break

        if conf.wudi_points_masked_by_eco_region is True:
            master_wudi = master_wudi.loc[(master_wudi.eco > 0)]
            size = np.array(master_wudi['pid']).size
            print('wudi_points_masked_by_eco_region size:', size)

        return master_wudi

    # eco_regions = parse_eco_regions()
    # util.save_asset(eco_regions, 'map-eco-regions')

    wudi_points = parse_wudi(new_wudi_netcdf, load.eco_regions(False))
    util.save_asset(wudi_points, 'wudi-points')

    output = {
        'wudi-points': wudi_points,
        'wudi-points-id-mask': list(wudi_points.pid.values)
    }

    util.save_asset(list(wudi_points.pid.values), 'wudi-points-indices')

    return output


def wudi_temporal_data_source(new_wudi_netcdf: os.path, force=False) -> dict:
    from cftime import num2date
    import netCDF4 as nC
    import numpy as np

    if conf.wudi_points_masked_by_eco_region is True:
        wudi_index_path = os.path.join(conf.support_path, 'wudi-points-indices-list.pkl')
        unmasked_indices_data = np.load(wudi_index_path, None, True)
    else:
        unmasked_indices_data = None

    ds = nC.Dataset(new_wudi_netcdf)

    #GET WUDI as day-up or day-down (1,-1) ARRAY or fallback to create it from 'ds' dataset
    def get_vectorized_wudi() -> np.ndarray:
        vectorized_file_path = os.path.join(conf.support_path, 'wudi-vectorized-ndarray.npy')
        if os.path.exists(vectorized_file_path) and force is False:
            vectorized = np.load(vectorized_file_path, None, True)
        else:

            def kool_aid(k):
                if k >= conf.wudi_UPWthr:
                    return 1.0
                elif k <= conf.wudi_DNWthr:
                    return -1.0
                else:
                    return 0.0

            if conf.wudi_points_masked_by_eco_region is True:
                source = ds['WUDI'][:, unmasked_indices_data]
            else:
                source = ds['WUDI'][:, :]

            print('get_vectorized_wudi np.vectorize ...', source.size)
            vectorized_kool = np.vectorize(kool_aid)
            vectorized = vectorized_kool(source)
            vectorized = np.ma.getdata(vectorized)
            util.save_asset(vectorized, 'wudi-vectorized')
        return vectorized

    #GET TIMES ARRAY or fallback to create it from 'ds' dataset
    def get_times() -> np.ndarray:
        times_file_path = os.path.join(conf.support_path, 'wudi-time-index-ndarray.npy')
        if os.path.exists(times_file_path) and force is False:
            valid_dates = np.load(times_file_path, None, True)
        else:
            times_static = []
            for n in range(ds['time'].size):
                n_date = num2date(ds['time'][n], units=ds['time'].units, calendar=ds['time'].calendar, has_year_zero=True)
                times_static.append([n_date.year, n_date.month, n_date.day])
                util.show_progress('parse_wudi_timeseries', n, ds['time'].size)
            valid_dates = np.array(times_static)
            util.save_asset(valid_dates, 'wudi-time-index')
        return valid_dates

    #GET EVENTS ARRAY or fallback to create it from 'ds' dataset
    def get_events_meta(source: np.ndarray) -> np.ndarray:
        events_file_path = os.path.join(conf.support_path, 'wudi-events-meta-ndarray.npy')
        if os.path.exists(events_file_path) and force is False:
            return np.load(events_file_path, None, True)
        else:
            def get_events(dailies, accumulator):
                rtx = None
                rtq = np.zeros(dailies.shape[1], dtype=np.int32)
                rng = len(dailies)
                for t in range(rng):
                    if t > 0:
                        rtq = np.add(rtq, rtx, where=rtx == dailies[t])
                        prc = np.where(rtx != dailies[t])
                        rtq[prc[0]] = 0
                        fin = np.where(np.abs(rtq) == conf.wudi_event_num_days - 1)
                        if fin[0].any():
                            accumulator[t] = fin
                    rtx = dailies[t]
                    util.show_progress('get_events_meta', t, rng)
                return accumulator

            kpt = np.ndarray((source.shape[0],), dtype=object)
            valid_events = get_events(source, kpt)
            util.save_asset(valid_events, 'wudi-events-meta')
            return valid_events

    times = get_times()
    daily = get_vectorized_wudi()
    events = get_events_meta(daily)

    block = {
        'times': times,
        'daily': daily,
        'events': events,
        'indices': unmasked_indices_data
    }

    return block


def make_wudi_points_database(args):
    import pandas as pd

    df = pd.read_pickle(os.path.join(conf.support_path, 'wudi-points-Dataframe.pkl'))
    df = df.applymap(util.cleaner_numeric, precision=5)

    # print(df.info())
    # exit()
    #
    # for n in range(df.shape[0]):
    #     print(list(df.iloc[n].values))

    dtypes = {
        'A_lat': 'REAL',
        'A_lon': 'REAL',
        'M_lat': 'REAL',
        'M_lon': 'REAL',
        'B_lat': 'REAL',
        'B_lon': 'REAL',
        'geo': 'INT',
        'eco': 'INT',
        'pid': 'INT'
    }

    conn = create_connection(conf.wudi_database_path)
    df.to_sql('wudi_points', conn, if_exists='replace', dtype=dtypes, index=False)
    conn.close()


def make_wudi_temporal_database(args):
    #//def new_parser(do_save_db: str = None, method: str = None, from_index: int = 0, wipe: str = None):
    #//aggregate is for 40-year view, built on top of derivative

    import numpy as np
    import netCDF4 as nC

    # do_save_db = 'save'
    # method = 'aggregate'  #'derivative'  #'aggregate'  #'daily'
    # wipe = None  #'yes'
    # from_index = 0

    basis_path = os.listdir(conf.source_path)[0]
    basis_path = os.path.join(conf.source_path, basis_path)
    print(basis_path)

    basis = wudi_temporal_data_source(basis_path, force=('force' in args))

    t_years = np.unique(basis['times'][:, 0])
    t_months = [e + 1 for e in range(12)]

    times = basis['times']
    daily = basis['daily']
    events = basis['events']
    indices = basis['indices']

    ds = nC.Dataset(basis_path)

    def event_fmt(evt, evt_month):
        lt = [str(times[evt[0]][1]).zfill(2), str(times[evt[0]][2]).zfill(2), str(evt[2] + 1)]
        if evt_month is not None:
            lt = lt[2:]
        return ''.join(lt)

    def event_fmt_aggregated(evt_index):
        return ''.join([str(times[evt_index][0]), str(times[evt_index][1]).zfill(2), str(times[evt_index][1]).zfill(2)])

    def observe_daily(d_index, d_width):
        d_time_record = ''.join([str(n).zfill(2) for n in times[d_index] if n is not None])
        d_events = events[d_index]
        some_arr = []
        for n in range(d_width):
            d_event = np.any(n == d_events[0]) if d_events is not None else False
            pid = int(indices[n])
            some_arr.append([
                int(d_time_record),
                pid,
                round(float(ds['WUDI'][d_index, pid]), conf.db_float_precision),
                int(d_event)])

        return some_arr

    def observe(obs_year, months):
        #// 1) run obs for year
        #// 2) run obs for year's months
        #// 3) return both
        return_object = []
        return_object_meta = []  #always 13

        def do_observation(o_year, o_month=None):
            if o_month:
                obs_indices = np.where((times[:, 0] == o_year) & (times[:, 1] == o_month))
            else:
                obs_indices = np.where(times[:, 0] == o_year)

            long = len(obs_indices[0])

            if long:
                ref = [o_year, o_month]
                time_record = ''.join([str(n).zfill(2) for n in ref if n is not None])
                wudi_at_time = daily[obs_indices[0], :]

                pos_sum = np.sum(wudi_at_time, where=wudi_at_time > 0, axis=0, dtype=np.int32)
                neg_sum = np.sum(wudi_at_time, where=wudi_at_time < 0, axis=0, dtype=np.int32)
                events_rst = [[i, events[i], n] for n, i in enumerate(obs_indices[0]) if events[i] is not None]

                up_max = np.nanmax(pos_sum)
                up_mean = np.nanmean(pos_sum, dtype=np.int32)
                down_max = np.nanmin(neg_sum)
                down_mean = np.nanmean(neg_sum, dtype=np.int32)
                return_object_meta.append([int(time_record), long, int(up_max), int(up_mean), int(down_max), int(down_mean)])

                #raw_values = ds['WUDI'][obs_indices[0], :]
                raw_values = ds['WUDI'][obs_indices[0], indices]
                up_max = np.nanmax(np.where(raw_values >= 0, raw_values, np.nan), axis=0)
                up_mean = np.nanmean(np.where(raw_values >= 0, raw_values, np.nan), axis=0)
                down_max = np.nanmin(np.where(raw_values <= 0, raw_values, np.nan), axis=0)
                down_mean = np.nanmean(np.where(raw_values <= 0, raw_values, np.nan), axis=0)

                for n in range(pos_sum.shape[0]):
                    pid = int(indices[n])
                    n_event = [event_fmt(i, o_month) for i in events_rst if np.any(n == i[1][0])]
                    n_e_ct = len(n_event)
                    n_e_ls = ','.join(n_event)
                    raw_data = [up_max[n], up_mean[n], down_max[n], down_mean[n]]
                    t_raw = ','.join([str(round(float(v), conf.db_float_precision)) for v in raw_data])
                    return_object.append([int(time_record), pid, int(pos_sum[n]), int(neg_sum[n]), t_raw, n_e_ct, n_e_ls])

        do_observation(obs_year)

        for m in months:
            do_observation(obs_year, m)

        return return_object, return_object_meta

    def observe_aggregate(d_width):
        return_object = []  #always 14
        return_object_meta = []  #always 13

        def get_events_aggregated():
            agg = np.ndarray((d_width,), dtype=object)
            for i in range(events.size):
                evt = events[i]
                if evt:
                    for e in evt[0]:
                        agg[e] = np.array([i], dtype=int) if agg[e] is None else np.append(agg[e], i)
                util.show_progress('get_events_aggregated', i, events.size)
            return agg

        events_aggregated = get_events_aggregated()

        wudi_at_time = daily[:, :]
        pos_sum = np.sum(wudi_at_time, where=wudi_at_time > 0, axis=0, dtype=np.int32)
        neg_sum = np.sum(wudi_at_time, where=wudi_at_time < 0, axis=0, dtype=np.int32)
        up_max = np.nanmax(pos_sum)
        up_mean = np.nanmean(pos_sum, dtype=np.int32)
        down_max = np.nanmin(neg_sum)
        down_mean = np.nanmean(neg_sum, dtype=np.int32)
        return_object_meta.append([40, events.size, int(up_max), int(up_mean), int(down_max), int(down_mean)])

        raw_values = ds['WUDI'][:, indices]
        up_max = np.nanmax(np.where(raw_values >= 0, raw_values, np.nan), axis=0)
        up_mean = np.nanmean(np.where(raw_values >= 0, raw_values, np.nan), axis=0)
        down_max = np.nanmin(np.where(raw_values <= 0, raw_values, np.nan), axis=0)
        down_mean = np.nanmean(np.where(raw_values <= 0, raw_values, np.nan), axis=0)

        for n in range(d_width):
            n_e_ct, n_e_ls = None, None
            pid = int(indices[n])
            if events_aggregated[n] is not None:
                n_event = [event_fmt_aggregated(i) for i in events_aggregated[n]]
                n_e_ct = len(n_event)
                n_e_ls = ','.join(n_event)
            raw_data = [up_max[n], up_mean[n], down_max[n], down_mean[n]]
            t_raw = ','.join([str(round(float(v), conf.db_float_precision)) for v in raw_data])
            return_object.append([40, pid, int(pos_sum[n]), int(neg_sum[n]), t_raw, n_e_ct, n_e_ls])

        return return_object, return_object_meta

    #SAVE simple traversal of raw dataset to database
    if do_save_db == 'save' and method is not None:
        def save_to_db(con: sqlite3.Connection, count: int, batch_size: int, index: int = 0, special: tuple = None):
            con.execute('BEGIN')
            try:
                for n in range(index, count):  #range(int(count / batch_size)):
                    if method == 'daily':
                        current_batch = observe_daily(n, batch_size)
                        con.executemany(f"INSERT INTO wudi_daily VALUES (?,?,?,?)", current_batch)
                        pass
                    if method == 'derivative':
                        current_batch, current_batch_meta = observe(special[0][n], special[1])
                        con.executemany(f"INSERT INTO wudi_derivative VALUES (?,?,?,?,?,?,?)", current_batch)
                        con.executemany(f"INSERT INTO wudi_derivative_meta VALUES (?,?,?,?,?,?)", current_batch_meta)
                        pass
                    if method == 'aggregate':
                        current_batch, current_batch_meta = observe_aggregate(batch_size)
                        con.executemany(f"INSERT INTO wudi_derivative VALUES (?,?,?,?,?,?,?)", current_batch)
                        con.execute(f"INSERT INTO wudi_derivative_meta VALUES (?,?,?,?,?,?)", current_batch_meta[0])
                        pass
                    if count > 1:
                        util.show_progress('obs', n, count)

            except KeyboardInterrupt:
                pass

            con.commit()

        conn = sqlite3.connect(conf.wudi_database_path, isolation_level=None)
        conn.execute('PRAGMA journal_mode = OFF;')
        conn.execute('PRAGMA synchronous = 0;')
        conn.execute('PRAGMA cache_size = 2000000;')  # give it 2 GB
        conn.execute('PRAGMA locking_mode = EXCLUSIVE;')
        conn.execute('PRAGMA temp_store = MEMORY;')

        if method == 'daily':
            acquire_table(conn, "wudi_daily", wipe)
            save_to_db(conn, count=daily.shape[0], batch_size=daily.shape[1], index=from_index)
            #save_to_db(conn, count=1, batch_size=600, index=from_index)
        elif method == 'derivative':
            acquire_table(conn, "wudi_derivative", wipe)
            acquire_table(conn, "wudi_derivative_meta", wipe)
            ct = len(t_years)
            bs = ct*13*daily.shape[1]
            save_to_db(conn, count=ct, batch_size=bs, index=from_index, special=(t_years, t_months,))
        elif method == 'aggregate':
            acquire_table(conn, "wudi_derivative", wipe)
            acquire_table(conn, "wudi_derivative_meta", wipe)
            save_to_db(conn, count=1, batch_size=daily.shape[1])
        else:
            print('no method selected.')
            return

    #PRINT index of date in timeseries YYYY,M,D
    if do_save_db == 'get_index':
        t_year, t_month, t_day = method.split(',')
        index = np.where((times[:, 0] == int(t_year)) & (times[:, 1] == int(t_month)) & (times[:, 2] == int(t_day)))
        print('new_parser get_index', method, index)

    #PRINT calls observe_aggregate
    if do_save_db == 'get_aggregate':
        observe_aggregate(daily.shape[1])
        pass


def set_source(args):
    fp = str(args[0])
    force_rebuild = True if len(args) > 1 else None

    if os.path.exists(fp):
        head_tail = os.path.split(fp)
        nc_target = os.path.join(conf.source_path, head_tail[1])
        shutil.copy(fp, nc_target)

        status = f"{str(datetime.now())}\n{nc_target} {os.path.getsize(nc_target)/1000}k\n"
        with open(os.path.join(conf.support_path, 'history.txt'), 'a+') as history:
            history.write(status)

        print(nc_target, 'was saved. building supporting files.')

        # 'wudi-points' , 'wudi-points-id-mask'
        wudi_points_result = wudi_geographical_data_source(nc_target, force_rebuild)
        print(wudi_points_result.keys())

        result = wudi_temporal_data_source(nc_target, force_rebuild)
        print(result.keys())



    else:
        print(fp, 'not found')


def build_databases(args):
    print('flags', args)

    make_wudi_points_database(args)
    make_wudi_temporal_database(args)

    pass


def run_test(args):
    pass
    # import netCDF4 as nC
    # import pandas as pd
    # import numpy as np
    #
    # wudi_path = os.path.join(conf.support_path, 'wudi-points-DataFrame.pkl')
    # df = pd.read_pickle(wudi_path)
    #
    # print(len(list(df.pid.values)))

    #
    # path = os.path.join(conf.source_path, args[0])
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # k_ind = df.loc[(df.eco > 0)]
    # mask = list(k_ind.index.values)
    #
    # #
    # ds = nC.Dataset(path)
    #
    # source = ds['WUDI'][:, mask]
    #
    # print(source, source.shape)
