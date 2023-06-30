import time
import sys
import numpy
import pickle
import glob

if __package__ is None:
    from pepid import blackboard
    from pepid import pepid_mp
else:
    from . import blackboard
    from . import pepid_mp

import logging
import os
import math
import tempfile
import struct

def run(cfg_file):
    """
    Entry point
    """

    if __package__ is None:
        from pepid import queries
        from pepid import db
        from pepid import search
    else:
        from . import queries
        from . import db
        from . import search

    log.info("Search phases to run: | " + ("Query Processing | " if blackboard.config['processing.query'].getboolean('enabled') else "") +
                                ("DB Processing | " if blackboard.config['processing.db'].getboolean('enabled') else "") +
                                ("Postprocessing | " if blackboard.config['postprocessing'].getboolean('enabled') else "") +
                                ("Score | " if blackboard.config['scoring'].getboolean('enabled') else ""))
    log.info("Preparing Input Databases...")
    db_paths = []
    if blackboard.config['processing.query'].getboolean('enabled'):
        db_paths.append(blackboard.DB_PATH + "_q.sqlite")
    if blackboard.config['processing.db'].getboolean('enabled'):
        db_paths.append(blackboard.DB_PATH + "_cands.sqlite")
    if blackboard.config['scoring'].getboolean('enabled'):
        db_paths.append(blackboard.DB_PATH + ".sqlite")
    for p in db_paths:
        if os.path.exists(p):
            os.remove(p)
    blackboard.prepare_connection()
    if blackboard.config['processing.query'].getboolean('enabled'):
        queries.prepare_db()
    if blackboard.config['processing.db'].getboolean('enabled'):
        db.prepare_db()
    blackboard.init_results_db()

    n_queries = -1
    base_path = blackboard.TMP_PATH
    log.info("Preparing Input Processing Nodes...")
    
    qnodes = blackboard.config['processing.query'].getint('workers')
    dbnodes = blackboard.config['processing.db'].getint('workers')
    snodes = blackboard.config['scoring'].getint('workers')

    if qnodes < 0 or dbnodes < 0 or snodes < 0:
        log.fatal("Node settings are query={}, db={}, search={}, but only values 0 or above allowed.".format(qnodes, dbnodes, snodes))
        sys.exit(-2)

    proc_spec = []

    if blackboard.config['processing.db'].getboolean('enabled'):
        batch_size = blackboard.config['processing.db'].getint('batch size')
        n_db = db.count_db()
        n_db_batches = math.ceil(n_db / batch_size)

        dbspecs = []

        # Have to use 2 separate steps to ensure that if there are overlaps, the decoys are dropped
        # The protein-reverse approach generates quite a few overlaps...
        dbspec = [(blackboard.here("db_node.py"), dbnodes, n_db_batches,
                                    [struct.pack("!cI{}sc".format(len(base_path)), bytes([0x00]), len(base_path), base_path.encode("utf-8"), "$".encode("utf-8")) for _ in range(dbnodes)],
                                    [struct.pack("!cQQI6sc", bytes([0x01]), b * batch_size, min((b+1) * batch_size, n_db), 6, "normal".encode('utf-8'), "$".encode("utf-8")) for b in range(n_db_batches)],
                                    [struct.pack("!cc", bytes([0x7f]), "$".encode('utf-8')) for _ in range(dbnodes)])]
        dbspecs.append(dbspec)

        if blackboard.config['processing.db'].getboolean('generate decoys'):
            dbdecoy_spec = [(blackboard.here("db_node.py"), dbnodes, n_db_batches,
                                        [struct.pack("!cI{}sc".format(len(base_path)), bytes([0x00]), len(base_path), base_path.encode("utf-8"), "$".encode("utf-8")) for _ in range(dbnodes)],
                                        [struct.pack("!cQQI5sc", bytes([0x01]), b * batch_size, min((b+1) * batch_size, n_db), 5, "decoy".encode('utf-8'), "$".encode("utf-8")) for b in range(n_db_batches)],
                                        [struct.pack("!cc", bytes([0x7f]), "$".encode('utf-8')) for _ in range(dbnodes)])]
            dbspecs.append(dbdecoy_spec)

        proc_spec = proc_spec
        for spec in dbspecs:
            proc_spec = proc_spec + spec

    if (blackboard.config['processing.query'].getboolean('enabled') or blackboard.config['postprocessing'].getboolean('queries')) or blackboard.config['scoring'].getboolean('enabled'):
        n_queries = queries.count_queries()

    if blackboard.config['processing.query'].getboolean('enabled'):
        batch_size = blackboard.config['processing.query'].getint('batch size')
        n_query_batches = math.ceil(n_queries / batch_size)

        qspec = [(blackboard.here("queries_node.py"), qnodes, n_query_batches,
                        [struct.pack("!cI{}sc".format(len(base_path)), bytes([0x00]), len(base_path), base_path.encode("utf-8"), "$".encode("utf-8")) for _ in range(qnodes)],
                        [struct.pack("!cQQc", bytes([0x01]), b * batch_size, min((b+1) * batch_size, n_queries), "$".encode("utf-8")) for b in range(n_query_batches)],
                        [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(qnodes)])] 

        proc_spec = proc_spec + qspec

    if len(proc_spec) != 0:
        pepid_mp.handle_nodes("Input Processing", proc_spec, cfg_file=cfg_file, tqdm_silence=tqdm_silence)

    if blackboard.config['postprocessing'].getboolean('enabled'):
        idx = 0

        qnodes = blackboard.config['processing.query'].getint('postprocessing workers')
        dbnodes = blackboard.config['processing.db'].getint('postprocessing workers')

        if qnodes < 0 or dbnodes < 0:
            log.fatal("Post-processing node settings are query={}, db={}, but only values 0 or above allowed.".format(qnodes, dbnodes))
            sys.exit(-2)

        specs = []

        if blackboard.config['postprocessing'].getboolean('db'):
            db_batch_size = blackboard.config['processing.db'].getint('batch size')
            n_db = db.count_peps()
            n_db_batches = math.ceil(n_db / db_batch_size)

            dbspec = [(blackboard.here("db_node.py"), dbnodes, n_db_batches,
                            [struct.pack("!cI{}sc".format(len(base_path)), bytes([0x00]), len(base_path), base_path.encode("utf-8"), "$".encode("utf-8")) for _ in range(dbnodes)],
                            [struct.pack("!cQQc", bytes([0x02]), b * db_batch_size, min((b+1) * db_batch_size, n_db), "$".encode('utf-8')) for b in range(n_db_batches)],
                            [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(dbnodes)])]
            specs.extend(dbspec)

        if blackboard.config['postprocessing'].getboolean('queries'):
            if n_queries < 0:
                n_queries = queries.count_queries()
            q_batch_size = blackboard.config['processing.query'].getint('batch size')
            n_query_batches = math.ceil(n_queries / q_batch_size)
            qspec = [(blackboard.here("queries_node.py"), qnodes, n_query_batches,
                            [struct.pack("!cI{}sc".format(len(base_path)), bytes([0x00]), len(base_path), base_path.encode("utf-8"), "$".encode("utf-8")) for _ in range(qnodes)],
                            [struct.pack("!cQQc", bytes([0x02]), b * q_batch_size, min((b+1) * q_batch_size, n_queries), "$".encode("utf-8")) for b in range(n_query_batches)],
                            [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(qnodes)])]
            specs.extend(qspec)

        if len(specs) != 0:
            pepid_mp.handle_nodes("Input Postprocessing", specs, cfg_file=cfg_file, tqdm_silence=tqdm_silence)

    cur = blackboard.CONN.cursor()
    blackboard.execute(cur, "CREATE INDEX IF NOT EXISTS c.cand_mass_idx ON candidates (mass ASC);")
    blackboard.execute(cur, "CREATE INDEX IF NOT EXISTS q.query_mass_idx ON queries (mass ASC);")
    del cur

    if blackboard.config['scoring'].getboolean('enabled'):
        batch_size = blackboard.config['scoring'].getint('batch size')
        n_search_batches = math.ceil(n_queries / batch_size)
        sspec = [(blackboard.here("search_node.py"), snodes, n_search_batches,
                        [struct.pack("!cI{}sc".format(len(base_path)), bytes([0x00]), len(base_path), base_path.encode("utf-8"), "$".encode("utf-8")) for _ in range(snodes)],
                        [struct.pack("!cQQc", bytes([0x01]), b * batch_size, min((b+1) * batch_size, n_queries), "$".encode("utf-8")) for b in range(n_search_batches)],
                        [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(snodes)])]

        pepid_mp.handle_nodes("Search", sspec, cfg_file=cfg_file, tqdm_silence=tqdm_silence)

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("USAGE: {} config.cfg".format(sys.argv[0]))
        sys.exit(-1)

    global cfg_file
    global log

    blackboard.config.read(blackboard.here(blackboard.here('data/default.cfg')))

    cfg_file = sys.argv[1]
    blackboard.config.read(cfg_file)

    tqdm_silence = blackboard.config['logging']['level'].lower() in ['fatal', 'error', 'warning']

    blackboard.TMP_PATH = os.path.join(blackboard.config['data']['tmpdir'], "pepidrun_" + next(tempfile._get_candidate_names()))
    blackboard.setup_constants() # overrides TMP_PATH if workdir setting points to a directory to reuse
    log = blackboard.LOG

    if(not os.path.exists(blackboard.TMP_PATH)):
        os.mkdir(blackboard.TMP_PATH)

    blackboard.LOCK = blackboard.acquire_lock()

    run(cfg_file)
