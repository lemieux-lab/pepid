import blackboard
import os
import time
import numpy
import tqdm
import queries

def write_output():
    """
    Exports search results to csv format as per the config settings.
    """

    max_cands = blackboard.config['search'].getint('max retained candidates')
    batch_size = blackboard.config['performance'].getint('batch size')

    out_fname = blackboard.config['data']['output']
    outf = open(out_fname, 'w')

    header = blackboard.RES_COLS[:-1] + ['rowid']

    outf.write("\t".join(header) + "\n")

    n_cands = blackboard.config['search'].getint('max retained candidates')
    n_queries = queries.count_queries()

    #cur = blackboard.CONN.cursor()
    import glob
    fname_pattern = list(filter(lambda x: len(x) > 0, blackboard.config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_*_pepidpart.sqlite"
    fname_path = os.path.join(blackboard.config['data']['tmpdir'], fname_pattern)

    files = glob.glob(fname_path)
    #blackboard.execute(cur, "SELECT {} FROM results as r1 WHERE r1.rowid IN (SELECT r2.rowid FROM results AS r2 WHERE r1.title = r2.title ORDER BY r2.score DESC LIMIT ?) ORDER BY title ASC, score DESC;".format(",".join(blackboard.RES_COLS)), (n_cands,))

    for f in tqdm.tqdm(files, total=len(files), desc="Dumping To CSV"):
        import sqlite3
        import time
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

        blackboard.execute(cur, "SELECT results.rowid, results.candrow, results.qrow, {} FROM results JOIN (SELECT title, IFNULL((SELECT score FROM results WHERE title = titles.title AND score > 0 ORDER BY score DESC LIMIT 1 OFFSET ?), -1) AS cutoff_score FROM (SELECT DISTINCT title FROM results) AS titles) AS cutoffs ON results.title = cutoffs.title AND results.score >= cutoffs.cutoff_score;".format(",".join(map(lambda x: "results." + x, blackboard.RES_COLS[:-3]))), (n_cands-1,))
        while True:
            results = cur.fetchmany(batch_size)
            results = list(map(dict, results))
            if len(results) == 0:
                break
            buff = ""
            res_data = []
            import search
            for r in results:
                blackboard.execute(main_cur, "SELECT * FROM candidates WHERE rowid = ? LIMIT 1;", (r['candrow'],))
                res_cand = main_cur.fetchone()
                blackboard.execute(main_cur, "SELECT * FROM queries WHERE rowid = ? LIMIT 1;", (r['qrow'],))
                res_q = main_cur.fetchone()
                res_data.append(search.crnhs([res_cand], res_q))
            for res, data in zip(results, res_data):
                fields = []
                for k in header[:-3]:
                    fields.append(str(res[k]).replace(";", ","))
                fields.append(str(data).replace(";", ","))
                buff = buff + "\t".join(fields) + "\n"
            outf.write(buff)
        del cur
        del conn

    outf.close()
    #results.close()
