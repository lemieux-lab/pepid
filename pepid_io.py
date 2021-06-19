import blackboard
import os
import sqlite3
import time

def write_output():
    """
    Exports search results to csv format as per the config settings.
    """
    results = None

    while results is None:
        try:
            results = sqlite3.connect(blackboard.CONN_STR.format(os.path.join(blackboard.TMP_PATH, "tmp.sqlite")), 1)
        except sqlite3.OperationalError as e:
            if e.args[0] == 'database is locked':
                time.sleep(1)
                continue
            else:
                raise e

    max_cands = blackboard.config['search'].getint('max retained candidates')

    out_fname = blackboard.config['data']['output']
    f = open(out_fname, 'w')

    cursor = results.cursor()
    cursor.execute("SELECT name FROM (PRAGMA_TABLE_INFO('results'));")
    header = [x[0] for x in cursor.fetchall()]

    f.write(";".join(header) + "\n")

    # Because we order scores DESC, there is no better score than score 0 if score 0 appears within the rank interval selected (i.e. no matches at or beyond that point).
    cursor.execute("SELECT {} FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY title ORDER BY score DESC) AS RANK FROM results) WHERE SCORE > 0 AND RANK <= ?;".format(",".join(header)), [max_cands])
    entries = cursor.fetchall()

    for res in entries:
        f.write(";".join(list(map(lambda x: str(x).replace(";", ","), res))) + "\n")
    f.close()

    results.close()

