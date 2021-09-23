import configparser
import os
import sqlite3
import time
import uuid

CONN = None
RES_CONN = None
config = configparser.ConfigParser(inline_comment_prefixes="#")
DB_TYPE = None
RES_TYPE = None
QUERY_TYPE = None
DB_COLS = None
RES_COLS = None
QUERY_COLS = None
DB_FNAME = None
RES_DB_FNAME = None
DB_PATH = None
RES_DB_PATH = None
KEY_DATA_DTYPE = None
LOCK = None
TMP_PATH = None

def create_table_str(table_name, table_cols, table_types, extra=[]):
    return "CREATE TABLE {} ({});".format(table_name, ",".join(list(map(lambda x: "{} {}".format(x[0], x[1]), zip(table_cols, table_types))) + extra))

def insert_all_str(table_name, table_cols):
    return "INSERT INTO {} ({}) VALUES ({});".format(table_name, ",".join(table_cols), ",".join(["?"]*len(table_cols)))

def maybe_insert_str(table_name, table_cols):
    return "INSERT OR IGNORE INTO {} ({}) VALUES ({});".format(table_name, ",".join(table_cols), ",".join(["?"]*len(table_cols)))

def select_str(table_name, table_cols, extra=""):
    return "SELECT {} FROM {} {};".format(",".join(table_cols), table_name, extra)

def init_results_db(generate=False, base_dir=None):
    global RES_DB_FNAME
    global RES_DB_PATH
    global RES_CONN

    if base_dir is None:
        base_dir = config['data']['tmpdir']

    if generate:
        unique = str(uuid.uuid4())
        RES_DB_FNAME = list(filter(lambda x: len(x) > 0, config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_{}_pepidpart.sqlite".format(unique)
        RES_DB_PATH = os.path.join(base_dir, RES_DB_FNAME)

    RES_CONN = None
    _CONN = None
    while RES_CONN is None:
        try:
            _CONN = sqlite3.connect("file:" + RES_DB_PATH + "?cache=shared", detect_types=1, uri=True, timeout=0.1)
            cur = _CONN.cursor()
            cur.execute("PRAGMA synchronous=OFF;")
            #cur.execute("PRAGMA mmap_size=8589934560;") # 8GB/48 threads
            #cur.execute("PRAGMA page_size=1638400;") # 16KB ~= size of cand entry
            #cur.execute("PRAGMA cache_size=100;")
            #cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA temp_store_directory='{}';".format(config['data']['tmpdir']))
            #cur.execute("PRAGMA journal_mode=WAL;")
            RES_CONN = _CONN
        except:
            if _CONN is not None:
                _CONN.close()
                _CONN = None
            time.sleep(0.1)
            continue

    cur = RES_CONN.cursor()
    execute(cur, create_table_str("main.results", RES_COLS, [t if ((i != RES_COLS.index("score")) or (not generate)) else (t + " CHECK(score > 0)") for i, t in enumerate(RES_TYPES)]))
    RES_CONN.commit()

def setup_constants():
    global RES_COLS
    global RES_TYPES
    global DB_COLS
    global DB_TYPES
    global QUERY_COLS
    global QUERY_TYPES

    global DB_FNAME
    global DB_PATH
    global RES_DB_FNAME
    global RES_DB_PATH

    RES_COLS = ["title", "desc", "seq", "modseq", "score", "data"]
    RES_TYPES = ["TEXT", "TEXT", "TEXT", "TEXT", "REAL", "BLOB"]

    DB_COLS = ["desc", "seq", "mods", "rt", "length", "mass", "spec", "meta"]
    DB_TYPES = ["TEXT", "TEXT", "BLOB", "REAL", "INTEGER", "REAL", "BLOB", "BLOB"]

    QUERY_COLS = ["title", "rt", "charge", "mass", "spec", "min_mass", "max_mass", "meta"]
    QUERY_TYPES = ["TEXT", "REAL", "INTEGER", "REAL", "BLOB", "REAL", "REAL", "BLOB"]

    DB_FNAME = list(filter(lambda x: len(x) > 0, config['data']['database'].split('/')))[-1].rsplit('.', 1)[0]
    RES_DB_FNAME = DB_FNAME
    DB_PATH = os.path.join(config['data']['tmpdir'], DB_FNAME)
    RES_DB_PATH = DB_PATH + ".sqlite"

    if not config['pipeline'].getboolean('db processing'):
        if config['data'].getboolean('preprocessed database', fallback=True):
            DB_PATH = config['data']['preprocessed database']
            RES_DB_PATH = DB_PATH + ".sqlite"
            DB_FNAME = list(filter(lambda x: len(x) > 0, config['data']['preprocessed database'].split('/')))[-1]
            RES_DB_FNAME = DB_FNAME

def prepare_connection():
    global CONN
    global RES_CONN
    CONN = None
    _CONN = None
    while CONN is None:
        try:
            import sys
            _CONN = sqlite3.connect("file:" + DB_PATH + ".sqlite?cache=shared", detect_types=1, uri=True, timeout=0.1)
            cur = _CONN.cursor()
            #cur.execute("PRAGMA threads=4;")
            cur.execute("PRAGMA synchronous=OFF;")
            #cur.execute("PRAGMA mmap_size=8589934560;") # 8GB/48 threads
            #cur.execute("PRAGMA mmap_size=8589934592;") # 8GB
            #cur.execute("PRAGMA page_size=1638400;") # 16KB ~= size of cand entry
            #cur.execute("PRAGMA cache_size=100;")
            #cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA temp_store_directory='{}';".format(config['data']['tmpdir']))
            #cur.execute("PRAGMA journal_mode=WAL;")

            cur.execute("ATTACH DATABASE ? AS c;", (DB_PATH + "_cands.sqlite",))
            cur.execute("ATTACH DATABASE ? AS q;", (DB_PATH + "_q.sqlite",))

            CONN = _CONN

        except:
            if _CONN is not None:
                _CONN.close()
                _CONN = None
            time.sleep(0.1)
            continue

    if RES_CONN is None:
        RES_CONN = CONN

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
