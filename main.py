#!/usr/bin/env python3
import utilities as util
from updater import process as updater_process
from builder import prepare
from builder import save_database
from builder import save_static
from builder import sectorize
from builder import process as builder_process
from tools import test_depth

import sys

command = {
    'set_source': updater_process.set_source,
    'show_support_file': util.show_support_file,
    'run_test': updater_process.run_test,
    'make_wudi_points_database': updater_process.make_wudi_points_database,
    'build_databases': updater_process.build_databases,
    'test_prep': prepare.geo_associations,
    'tests': save_database.tests,
    'static_tests': save_static.tests,
    'places': prepare.tests,
    'sectorize': sectorize.save_sectors,
    'post_process': builder_process.post,
    'day_test': test_depth
}


def update_wudi_points():
    #//don't forget to automate addition of indices to tables!
    #//establish command order here
    pass


if __name__ == '__main__':
    print('test {0}'.format(updater_process.test()))

    if sys.argv[1] in command.keys():
        arg_special = sys.argv[2:]
        if len(arg_special) > 0:
            command[sys.argv[1]](*arg_special)
        else:
            command[sys.argv[1]]()