import configparser
import os
import sqlite3
import time
import uuid
import msgpack
import sys
import fcntl

import logging

import subprocess as sp

LOG = None
CONN = None
RES_CONN = None
META_CONN = None
LOCK = None

config = configparser.ConfigParser(inline_comment_prefixes="#")

DB_TYPES = None
RES_TYPES = None
META_TYPES = None
QUERY_TYPES = None
DB_COLS = None
RES_COLS = None
META_COLS = None
QUERY_COLS = None
DB_FNAME = None
RES_DB_FNAME = None
META_DB_FNAME = None
DB_PATH = None
RES_DB_PATH = None
META_DB_PATH = None
TMP_PATH = None

# Simple wrapper to hook into the sqlite3 auto-conversion system...
class Spectrum(object):
    def __init__(self, x):
        self.data = x

class Meta(object):
    def __init__(self, x):
        self.data = x

def create_table_str(table_name, table_cols, table_types, extra=[]):
    return "CREATE TABLE IF NOT EXISTS {} ({});".format(table_name, ",".join(list(map(lambda x: "{} {}".format(x[0], x[1]), zip(table_cols, table_types))) + extra))

def insert_all_str(table_name, table_cols):
    return "INSERT INTO {} ({}) VALUES ({});".format(table_name, ",".join(table_cols), ",".join(["?"]*len(table_cols)))

def insert_dict_str(table_name, table_cols):
    return "INSERT INTO {} ({}) VALUES ({});".format(table_name, ",".join(table_cols), ",".join([":" + x for x in table_cols]))

def insert_dict_extra_str(table_name, table_cols, extra):
    return "INSERT INTO {} ({}) VALUES ({}) {};".format(table_name, ",".join(table_cols), ",".join([":" + x for x in table_cols]), extra)

def maybe_insert_dict_extra_str(table_name, table_cols, extra):
    return "INSERT OR IGNORE INTO {} ({}) VALUES ({}) {};".format(table_name, ",".join(table_cols), ",".join([":" + x for x in table_cols]), extra)

def maybe_insert_str(table_name, table_cols):
    return "INSERT OR IGNORE INTO {} ({}) VALUES ({});".format(table_name, ",".join(table_cols), ",".join(["?"]*len(table_cols)))

def maybe_insert_dict_str(table_name, table_cols):
    return "INSERT OR IGNORE INTO {} ({}) VALUES ({});".format(table_name, ",".join(table_cols), ",".join([":" + x for x in table_cols]))

def select_str(table_name, table_cols, extra=""):
    return "SELECT {} FROM {} {};".format(",".join(table_cols), table_name, extra)

def init_results_db(generate=False, base_dir=None):
    global RES_DB_FNAME
    global RES_DB_PATH
    global META_DB_FNAME
    global META_DB_PATH
    global RES_CONN
    global META_CONN

    if base_dir is None:
        base_dir = TMP_PATH

    if generate:
        unique = str(uuid.uuid4())
        RES_DB_FNAME = list(filter(lambda x: len(x) > 0, config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_{}_pepidpart".format(unique)
        META_DB_FNAME = RES_DB_FNAME + "_meta.sqlite"
        RES_DB_FNAME = RES_DB_FNAME + ".sqlite"
        RES_DB_PATH = os.path.join(base_dir, RES_DB_FNAME)
        META_DB_PATH = os.path.join(base_dir, META_DB_FNAME)

    RES_CONN = None
    _CONN = None
    while RES_CONN is None:
        try:
            _CONN = sqlite3.connect("file:" + RES_DB_PATH + "?cache=shared", detect_types=1, uri=True, timeout=0.1)
            cur = _CONN.cursor()
            cur.execute("PRAGMA synchronous=OFF;")
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

    RES_CONN.row_factory = sqlite3.Row
    cur = RES_CONN.cursor()
    execute(cur, create_table_str("main.results", RES_COLS, RES_TYPES))
    RES_CONN.commit()

    META_CONN = None
    _CONN = None
    while META_CONN is None:
        try:
            _CONN = sqlite3.connect("file:" + META_DB_PATH + "?cache=shared", detect_types=1, uri=True, timeout=0.1)
            cur = _CONN.cursor()
            cur.execute("PRAGMA synchronous=OFF;")
            #cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA temp_store_directory='{}';".format(config['data']['tmpdir']))
            #cur.execute("PRAGMA journal_mode=WAL;")
            META_CONN = _CONN
        except:
            if _CONN is not None:
                _CONN.close()
                _CONN = None
            time.sleep(0.1)
            continue

    META_CONN.row_factory = sqlite3.Row
    cur = META_CONN.cursor()
    execute(cur, create_table_str("main.meta", META_COLS, META_TYPES))
    META_CONN.commit()

def setup_constants():
    global RES_COLS
    global META_COLS
    global RES_TYPES
    global META_TYPES
    global DB_COLS
    global DB_TYPES
    global QUERY_COLS
    global QUERY_TYPES

    global DB_FNAME
    global DB_PATH
    global RES_DB_FNAME
    global RES_DB_PATH
    global META_DB_FNAME
    global META_DB_PATH

    global TMP_PATH

    global LOG

    RES_COLS = ["title", "desc", "seq", "modseq", "score", "query_charge", "query_mass", "cand_mass", "candrow", "qrow", "file", "rrow"]
    RES_TYPES = ["TEXT", "TEXT", "TEXT", "BLOB", "REAL", "INTEGER", "REAL", "REAL", "INTEGER", "INTEGER", "TEXT", "INTEGER"]

    META_COLS = ["qrow", "candrow", "data", "extra", "score", "rrow"] # score is used to mirror insertion exclusion via CHECK(score > 0) from the 'data' db
    META_TYPES = ["INTEGER", "INTEGER", "BLOB", "BLOB", "REAL", "INTEGER"]

    DB_COLS = ["desc", "decoy", "rt", "length", "mass", "seq", "mods", "spec", "meta"]
    DB_TYPES = ["TEXT", "INTEGER", "REAL", "INTEGER", "REAL", "TEXT", "AUTOBLOB", "SPECTRUM", "META"]

    QUERY_COLS = ["title", "rt", "charge", "mass", "spec", "min_mass", "max_mass", "meta"]
    QUERY_TYPES = ["TEXT", "REAL", "INTEGER", "REAL", "SPECTRUM", "REAL", "REAL", "META"]

    sqlite3.register_adapter(Spectrum, lambda x: msgpack.dumps(x.data))
    sqlite3.register_adapter(Meta, lambda x: msgpack.dumps(x.data))
    sqlite3.register_converter("spectrum", lambda x: Spectrum(msgpack.loads(x)))
    sqlite3.register_converter("meta", lambda x: Meta(msgpack.loads(x)))
    sqlite3.register_converter("autoblob", lambda x: msgpack.loads(x))

    DB_FNAME = list(filter(lambda x: len(x) > 0, config['data']['database'].split('/')))[-1].rsplit('.', 1)[0]
    RES_DB_FNAME = DB_FNAME + ".sqlite"
    META_DB_FNAME = DB_FNAME + "_meta.sqlite"

    workdir = config['data']['workdir']
    try:
        no_workdir = config['data'].getboolean('workdir')
        if no_workdir:
            sys.stderr.write("[pepid]: workdir set to true is invalid; expected path or false\n")
            sys.exit(-1)
    except:
        TMP_PATH = config['data']['workdir']

    DB_PATH = os.path.join(TMP_PATH, DB_FNAME)
    RES_DB_PATH = DB_PATH + ".sqlite"
    META_DB_PATH = DB_PATH + "_meta.sqlite"

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    LOG = logging.getLogger("pepid")
    LOG.setLevel(config['logging']['level'].upper())
    LOG.addHandler(handler)

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
            cur.execute("PRAGMA synchronous=OFF;")
            #cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA temp_store_directory='{}';".format(config['data']['tmpdir']))
            #cur.execute("PRAGMA journal_mode=WAL;")

            cur.execute("ATTACH DATABASE ? AS c;", (DB_PATH + "_cands.sqlite",))
            cur.execute("ATTACH DATABASE ? AS q;", (DB_PATH + "_q.sqlite",))

            CONN = _CONN
            CONN.row_factory = sqlite3.Row

        except:
            if _CONN is not None:
                _CONN.close()
                _CONN = None
            time.sleep(0.1)
            continue

    if RES_CONN is None:
        RES_CONN = CONN

def execute(cur, *args):
    import sys
    while True:
        try:
            ret = cur.execute(*args)
            return ret
        except Exception as e:
            time.sleep(0.1)
            continue

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

def subprocess(args):
    extra_args = list(filter(lambda x: len(x) != 0, config['performance']['extra args'].strip().split(" ")))
    args = list(filter(lambda x: len(x) != 0, args))
    payload = [config['performance']['python']] + extra_args + args
    proc = sp.Popen(payload)
    return proc

def lock():
    fcntl.lockf(LOCK, fcntl.LOCK_EX)

def unlock():
    fcntl.lockf(LOCK, fcntl.LOCK_UN)

def here(path):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
