import os
import numpy
import time
import pickle
import sqlite3

# This is an example script to show how a 'multistaged' pepid operation mode might work.
# Two config files (example_proteometools_stage1.cfg and example_proteometools_stage2.cfg) are provided.
# To operate the pipeline, one may do `python -mpepid proteometools_stage1.cfg && /path/to/pepid_mgf_meta.py && python -mpepid proteometools_stage2.cfg`, for example.
# (after copying, renaming, and fixing the parameters in, the example configs).
# This operation mode was used to generate some fdp measures for pepid validation with great success.

if __package__ is None or __package__ == '':
    import blackboard
else:
    from . import blackboard

def insert_mgf_field(cfg, key):
    blackboard.config.read(blackboard.here("data/default.cfg"))
    blackboard.config.read(cfg)
    blackboard.setup_constants()

    mf = blackboard.config['data']['queries']
    f = open(mf, 'r')

    found = False
    title = None
    val = None

    mgf = {}

    for l in f:
        if l.startswith('BEGIN IONS'):
            found = False
            title = None
            val = None
            continue
        if l.startswith('END IONS'):
            if not found:
                blackboard.LOG.error("Could not find key '{}' (failed at entry titled '{}')".format(key, title))
                sys.exit(-2)
            mgf[title] = val
            continue
        if l.startswith('TITLE='):
            title = l.strip()[len('TITLE='):]
            continue
        if l.startswith(key + "="):
            val = l.strip()[len(key)+1:]
            found = True
            continue
        else:
            continue

    qdb = os.path.join(blackboard.config['data']['workdir'], blackboard.config['data']['database'].split('/')[-1].rsplit('.', 1)[0] + "_q.sqlite")
    conn = sqlite3.connect("file:" + qdb + "?cache=shared", detect_types=1, uri=True,)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    update_cur = conn.cursor()
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store_directory='{}';".format(blackboard.config['data']['tmpdir']))

    cur.execute("SELECT rowid, * FROM queries ORDER BY rowid;")

    ret = []

    while True:
        queries = cur.fetchmany(620000)
        if len(queries) == 0:
            break

        for query in queries:
            query = dict(query)
            extra = msgpack.loads(query['meta'])
            if extra is None:
                extra = {}
            extra['mgf:' + key] = mgf[query['title']]
            ret.append({'rowid': query['rowid'], 'data': msgpack.dumps(extra)})

        update_cur.executemany("UPDATE queries SET meta=:data WHERE rowid=:rowid;", ret)
        conn.commit()
        ret = []

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("USAGE: {} config.cfg <mgf field>".format(sys.argv[0]))

    insert_mgf_field(sys.argv[1], sys.argv[2])
