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

    for f in tqdm.tqdm(files, total=len(files), desc="Dumping To CSV"):
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

        blackboard.execute(cur, "SELECT results.rowid, {} FROM results JOIN (SELECT qrow, IFNULL((SELECT score FROM results WHERE qrow = qrows.qrow ORDER BY score DESC LIMIT 1 OFFSET ?), -1) AS cutoff_score FROM (SELECT DISTINCT qrow FROM results) AS qrows) AS cutoffs ON results.qrow = cutoffs.qrow AND results.score >= cutoffs.cutoff_score ORDER BY results.title ASC, results.score DESC;".format(",".join(map(lambda x: "results." + x, header))), (max_cands-1,))
        fetch_batch_size = min(batch_size, 62000) # The maximum batch size supported by the default sqlite engine is a bit more than 62000
        while True:
            results = cur.fetchmany(fetch_batch_size)
            if len(results) == 0:
                break

            results = list(map(dict, results))
            for idata, data in enumerate(results):
                buff = ""
                fields = []
                for k in header:
                    fields.append(str(data[k]).replace("\t", "    "))
                #fields.append(str(data['rowid']))
                buff = buff + "\t".join(fields) + "\n"
                outf.write(buff)
        del cur
        del conn

    outf.close()
    #results.close()
