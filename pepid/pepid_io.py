import blackboard
import os
import time
import numpy
import pickle
import tqdm
import queries

def write_output():
    """
    Exports search results to csv format as per the config settings.
    """

    import sys
    sys.stderr.write("write_ouptut: start!\n")
    max_cands = blackboard.config['search'].getint('max retained candidates')
    batch_size = blackboard.config['performance'].getint('batch size')

    out_fname = blackboard.config['data']['output']
    f = open(out_fname, 'w')

    header = blackboard.RES_COLS

    sys.stderr.write("Write header: {}\n".format(header))
    f.write("\t".join(header) + "\n")

    n_cands = blackboard.config['search'].getint('max retained candidates')
    n_queries = queries.count_queries()

    cur = blackboard.CONN.cursor()
    sys.stderr.write("get request starts! {} total queries...\n".format(n_queries))
    blackboard.execute(cur, "SELECT {} FROM results as r1 WHERE r1.rowid IN (SELECT r2.rowid FROM results AS r2 WHERE r1.title = r2.title ORDER BY r2.score DESC LIMIT ?);".format(",".join(blackboard.RES_COLS)), (n_cands,))
    sys.stderr.write("Query str for output is \"{}\"\n".format("SELECT {} FROM results as r1 WHERE r1.rowid IN (SELECT r2.rowid FROM results AS r2 WHERE r1.title = r2.title ORDER BY r2.score DESC LIMIT ?);".format(",".join(blackboard.RES_COLS))))

    data_idx = blackboard.RES_COLS.index("data")
    progress = tqdm.tqdm(desc='Output Top Results', total=n_queries * n_cands)
    while True:
        sys.stderr.write("GOING TO FETCH {}\n".format(batch_size))
        results = cur.fetchmany(batch_size)
        sys.stderr.write("FETCH {} -> {}\n".format(batch_size, len(results)))
        if len(results) == 0:
            break
        for res in results:
            for i, r in enumerate(res):
                if i == data_idx:
                    r = pickle.loads(r)
                f.write(str(r).replace(";", ","))
                if i < len(res)-1:
                    f.write("\t")
            f.write("\n")
            progress.update(1)
    progress.close()
    f.close()

    #results.close()
