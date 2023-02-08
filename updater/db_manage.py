import sqlite3
from sqlite3 import Error
import config as conf


def acquire_table(conn: sqlite3.Connection, name: str, wipe: str = None):
    #get the count of tables with the name
    c = conn.cursor()
    c.execute(""" SELECT count(name) FROM sqlite_master WHERE type='table' AND name='%s'; """ % name)
    if c.fetchone()[0] == 1:
        if wipe is not None:
            print(f'"{name}" table exists, cleaning.')
            c.execute(""" DELETE FROM %s; """ % name)
            conn.commit()
    else:
        conn.execute(conf.db_tables[name])
        print(f'"{name}" table created.')


def create_connection(db_file) -> sqlite3.Connection:
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print('connected', sqlite3.version)
    except Error as e:
        print(e)
    return conn

