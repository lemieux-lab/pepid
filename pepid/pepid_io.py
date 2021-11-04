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
    f = open(out_fname, 'w')

    header = blackboard.RES_COLS

    f.write("\t".join(header) + "\n")

    n_cands = blackboard.config['search'].getint('max retained candidates')
    n_queries = queries.count_queries()

    cur = blackboard.CONN.cursor()
    #blackboard.execute(cur, "SELECT {} FROM results as r1 WHERE r1.rowid IN (SELECT r2.rowid FROM results AS r2 WHERE r1.title = r2.title ORDER BY r2.score DESC LIMIT ?) ORDER BY title ASC, score DESC;".format(",".join(blackboard.RES_COLS)), (n_cands,))
    blackboard.execute(cur, "SELECT {} FROM results JOIN (SELECT title, IFNULL((SELECT score FROM results WHERE title = titles.title AND score > 0 ORDER BY score DESC LIMIT 1 OFFSET ?), -1) AS cutoff_score FROM (SELECT DISTINCT title FROM results) AS titles) AS cutoffs ON results.title = cutoffs.title AND results.score >= cutoffs.cutoff_score;".format(",".join(map(lambda x: "results." + x, blackboard.RES_COLS))), (n_cands-1,))

    data_idx = blackboard.RES_COLS.index("data")
    progress = tqdm.tqdm(desc='Output Top Results', total=n_queries * n_cands)
    while True:
        results = cur.fetchmany(batch_size)
        if len(results) == 0:
            break
        for res in results:
            for i, r in enumerate(res):
                f.write(str(r).replace(";", ",") if i < len(res)-1 else str(r.data))
                if i < len(res)-1:
                    f.write("\t")
            f.write("\n")
            progress.update(1)
    progress.close()
    f.close()

    #results.close()
