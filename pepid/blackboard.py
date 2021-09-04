import configparser
import os
import sqlite3
import time

CONN = None
config = configparser.ConfigParser(inline_comment_prefixes="#")
DB_TYPE = None
RES_TYPE = None
QUERY_TYPE = None
DB_COLS = None
RES_COLS = None
QUERY_COLS = None
DB_FNAME = None
DB_PATH = None
KEY_DATA_DTYPE = None
LOCK = None

def create_table_str(table_name, table_cols, table_types, extra=[]):
    return "CREATE TABLE {} ({});".format(table_name, ",".join(list(map(lambda x: "{} {}".format(x[0], x[1]), zip(table_cols, table_types))) + extra))

def insert_all_str(table_name, table_cols):
    return "INSERT INTO {} ({}) VALUES ({});".format(table_name, ",".join(table_cols), ",".join(["?"]*len(table_cols)))

def maybe_insert_str(table_name, table_cols):
    return "INSERT OR IGNORE INTO {} ({}) VALUES ({});".format(table_name, ",".join(table_cols), ",".join(["?"]*len(table_cols)))

def select_str(table_name, table_cols, extra=""):
    return "SELECT {} FROM {} {};".format(",".join(table_cols), table_name, extra)

def setup_constants():
    global RES_COLS
    global RES_TYPES
    global DB_COLS
    global DB_TYPES
    global QUERY_COLS
    global QUERY_TYPES

    global DB_FNAME
    global DB_PATH

    RES_COLS = ["title", "desc", "score", "data"]
    RES_TYPES = ["TEXT", "TEXT", "REAL", "BLOB"]

    DB_COLS = ["desc", "seq", "mods", "rt", "length", "mass", "spec"]
    DB_TYPES = ["TEXT", "TEXT", "BLOB", "REAL", "INTEGER", "REAL", "BLOB"]

    QUERY_COLS = ["title", "rt", "charge", "mass", "spec", "min_mass", "max_mass", "meta"]
    QUERY_TYPES = ["TEXT", "REAL", "INTEGER", "REAL", "BLOB", "REAL", "REAL", "BLOB"]

    DB_FNAME = list(filter(lambda x: len(x) > 0, config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + ".sqlite"
    DB_PATH = os.path.join(config['data']['tmpdir'], DB_FNAME)

    if not config['pipeline'].getboolean('db processing'):
        if config['data'].getboolean('preprocessed database', fallback=True):
            DB_PATH = config['data']['preprocessed database']
            DB_FNAME = list(filter(lambda x: len(x) > 0, config['data']['preprocessed database'].split('/')))[-1]

def prepare_connection():
    global CONN
    CONN = None
    _CONN = None
    while CONN is None:
        try:
            _CONN = sqlite3.connect("file:" + DB_PATH + "?cache=shared", detect_types=1, uri=True, timeout=0.1)
            cur = _CONN.cursor()
            cur.execute("PRAGMA threads=4;")
            cur.execute("PRAGMA synchronous=OFF;")
            cur.execute("PRAGMA mmap_size=8589934592;") # 8GB
            cur.execute("PRAGMA page_size=1638400;") # 16KB ~= size of cand entry
            cur.execute("PRAGMA cache_size=100;")
            cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA journal_mode=OFF;")
            CONN = _CONN
        except:
            if _CONN is not None:
                _CONN.close()
                _CONN = None
            time.sleep(0.1)
            continue



def execute(cur, *args):
    while True:
        try:
            ret = cur.execute(*args)
        except:
            time.sleep(0.1)
            continue
        return ret

def executemany(cur, *args):
    while True:
        try:
            ret = cur.executemany(*args)
        except:
            time.sleep(0.1)
            continue
        return ret

def commit():
    while True:
        try:
            CONN.commit()
        except:
            time.sleep(0.1)
            continue
        break
