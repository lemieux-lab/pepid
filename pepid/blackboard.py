import configparser
import os

CONN_STR = "{}"
config = configparser.ConfigParser(inline_comment_prefixes="#")
DB_DTYPE = None
RES_DTYPE = None
QUERY_DTYPE = None
DB_FNAME = None
DB_PATH = None
KEY_DATA_DTYPE = None

def setup_constants():
    global DB_DTYPE
    global RES_DTYPE
    global QUERY_DTYPE
    global DB_FNAME
    global DB_PATH
    global KEY_DATA_DTYPE

    RES_DTYPE = [('title', 'unicode', 1024), ('description', 'unicode', 1024), ('seq', 'unicode', 128), ('modseq', 'float32', 128), ('calc_mass', 'float32'), ('mass', 'float32'), ('rt', 'float32'), ('charge', 'int32'), ('length', 'int32'), ('score', 'float64'), ('score_data', 'unicode', 10240)]
    DB_DTYPE = [('description', 'unicode', 1024), ('sequence', 'unicode', 128), ('mods', 'float32', 128), ('rt', 'float32'), ('length', 'int32'), ('mass', 'float32'), ('npeaks', 'int32'), ('spec', 'float32', (config['search'].getint('max peaks'), 2)), ('meta', 'unicode', 10240)]
    QUERY_DTYPE = [('title', 'unicode', 1024), ('rt', 'float32'), ('charge', 'int32'), ('mass', 'float32'), ('spec', 'float32', (config['search'].getint('max peaks'), 2)), ('npeaks', 'int32'), ('min_mass', 'float32'), ('max_mass', 'float32'), ('meta', 'unicode', 10240)]
    KEY_DATA_DTYPE = [('desc', 'unicode', 1024), ('seq', 'unicode', 128), ('mods', 'float32', 128)]

    DB_FNAME = list(filter(lambda x: len(x) > 0, config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + ".bin"
    DB_PATH = os.path.join(config['data']['tmpdir'], DB_FNAME)

    if not config['pipeline'].getboolean('db processing'):
        if config['data'].getboolean('preprocessed database', fallback=True):
            DB_PATH = config['data']['preprocessed database']
            DB_FNAME = list(filter(lambda x: len(x) > 0, config['data']['preprocessed database'].split('/')))[-1]
