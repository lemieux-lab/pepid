import os
import time
import numpy
import sys
import sqlite3

import struct
import glob
import math

if __package__ is None or __package__ == '':
    import blackboard
    import queries
else:
    from . import blackboard
    from . import queries

def write_output(start, end):
    """
    Exports search results to tsv format as per the config settings.
    """

    max_cands = blackboard.config['output'].getint('max retained candidates')

    import glob
    fname_pattern = list(filter(lambda x: len(x) > 0, blackboard.config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_*_pepidpart.sqlite"
    fname_path = os.path.join(blackboard.TMP_PATH, fname_pattern)

    files = sorted(glob.glob(fname_path))[start:end]
    if len(files) == 0:
        return

    out_fname = blackboard.config['data']['output']

    header = blackboard.RES_COLS

    import collections
    counts = collections.defaultdict(int)

    for f in files:
        while True:
            try:
                conn = sqlite3.connect("file:{}?cache=shared".format(f), detect_types=1, uri=True, timeout=0.1)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                main_cur = blackboard.CONN.cursor()
                blackboard.execute(cur, "PRAGMA synchronous=OFF;")
                blackboard.execute(cur, "PRAGMA temp_store_directory='{}';".format(blackboard.config['data']['tmpdir']))
                break
            except:
                time.sleep(1)
                continue

        blackboard.execute(cur, "SELECT results.rowid, {} FROM results JOIN (SELECT qrow, IFNULL((SELECT score FROM results WHERE qrow = qrows.qrow ORDER BY score DESC LIMIT 1 OFFSET ?), -1) AS cutoff_score FROM (SELECT DISTINCT qrow FROM results) AS qrows) AS cutoffs ON results.qrow = cutoffs.qrow AND results.score >= cutoffs.cutoff_score ORDER BY results.qrow ASC, results.score DESC;".format(",".join(map(lambda x: "results." + x, header))), (max_cands-1,))
        fetch_batch_size = 62000 # The maximum batch size supported by the default sqlite engine is a bit more than 62000

        while True:
            results = cur.fetchmany(fetch_batch_size)
            if len(results) == 0:
                break

            blackboard.lock()
            outf = open(out_fname, 'a')

            for idata, data in enumerate(results):
                data = dict(data)
                # Need this second counter because if we have more than X results with the same score
                # we end up grabbing it all anyway. Example:
                # scores 1 1 1 1 1 1 1 1 1 1 1 0.9 0.9 0.9 with top 10:
                #   We end up selecting (>=) 11 1's instead of the max 10.
                if counts[data['qrow']] >= max_cands:
                    continue
                else:
                    fields = []
                    for k in header:
                        fields.append(str(data[k]).replace("\t", "    "))
                    #fields.append(str(data['rowid']))
                    outf.write("\t".join(fields) + "\n")
                    counts[data['qrow']] += 1

            outf.close()
            blackboard.unlock()

        del cur
        del conn

if __name__ == '__main__':
    blackboard.config.read(blackboard.here("data/default.cfg"))
    blackboard.config.read(sys.argv[1])

    cfg_file = sys.argv[1]

    blackboard.setup_constants()

    if __package__ is None or __package__ == '':
        import pepid_mp 
    else:
        from . import pepid_mp

    log_level = blackboard.config['logging']['level'].lower()

    out_fname = blackboard.config['data']['output']

    header = blackboard.RES_COLS

    outf = open(out_fname, 'w')
    outf.write("\t".join(header) + "\n")
    outf.close()

    nworkers = blackboard.config['output'].getint('workers')
    batch_size = blackboard.config['output'].getint('batch size')

    fname_pattern = list(filter(lambda x: len(x) > 0, blackboard.config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_*_pepidpart.sqlite"
    fname_path = os.path.join(blackboard.TMP_PATH, fname_pattern)
    all_files = glob.glob(fname_path)

    n_total = len(all_files)

    n_batches = math.ceil(n_total / batch_size)
    spec = [(blackboard.here("output_node.py"), nworkers, n_batches,
                    [struct.pack("!cI{}sc".format(len(blackboard.TMP_PATH)), bytes([0x00]), len(blackboard.TMP_PATH), blackboard.TMP_PATH.encode("utf-8"), "$".encode("utf-8")) for _ in range(nworkers)],
                    [struct.pack("!cQQc", bytes([0x01]), b * batch_size, min((b+1) * batch_size, n_total), "$".encode("utf-8")) for b in range(n_batches)],
                    [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(nworkers)])]

    pepid_mp.handle_nodes("Output TSV", spec, cfg_file=cfg_file, tqdm_silence=log_level in ['fatal', 'error', 'warning'])
