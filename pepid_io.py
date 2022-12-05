import blackboard
import os
import time
import numpy
import tqdm
import queries
import sys
import sqlite3

def write_output():
    """
    Exports search results to csv format as per the config settings.
    """

    max_cands = blackboard.config['output'].getint('max retained candidates')
    batch_size = blackboard.config['output'].getint('batch size')

    out_fname = blackboard.config['data']['output']
    outf = open(out_fname, 'w')

    header = blackboard.RES_COLS

    outf.write("\t".join(header) + "\n")

    n_queries = queries.count_queries()

    import glob
    fname_pattern = list(filter(lambda x: len(x) > 0, blackboard.config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_*_pepidpart.sqlite"
    fname_path = os.path.join(blackboard.TMP_PATH, fname_pattern)

    files = glob.glob(fname_path)

    import collections
    counts = collections.defaultdict(int)

    for f in tqdm.tqdm(files, total=len(files), desc="Dumping To TSV"):
        while True:
            try:
                conn = sqlite3.connect("file:{}?cache=shared".format(f), detect_types=1, uri=True, timeout=0.1)
                break
            except:
                time.sleep(1)
                continue
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        main_cur = blackboard.CONN.cursor()
        blackboard.execute(cur, "PRAGMA synchronous=OFF;")
        blackboard.execute(cur, "PRAGMA temp_store_directory='{}';".format(blackboard.config['data']['tmpdir']))

        blackboard.execute(cur, "SELECT results.rowid, {} FROM results JOIN (SELECT qrow, IFNULL((SELECT score FROM results WHERE qrow = qrows.qrow ORDER BY score DESC LIMIT 1 OFFSET ?), -1) AS cutoff_score FROM (SELECT DISTINCT qrow FROM results) AS qrows) AS cutoffs ON results.qrow = cutoffs.qrow AND results.score >= cutoffs.cutoff_score ORDER BY results.qrow ASC, results.score DESC;".format(",".join(map(lambda x: "results." + x, header))), (max_cands-1,))
        fetch_batch_size = min(batch_size, 62000) # The maximum batch size supported by the default sqlite engine is a bit more than 62000

        while True:
            results = cur.fetchmany(fetch_batch_size)
            if len(results) == 0:
                break

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
        del cur
        del conn

    outf.close()
    #results.close()
