import blackboard
import sqlite3
import os

def post_process():
    results = sqlite3.connect(blackboard.CONN_STR.format(os.path.join(blackboard.TMP_PATH, "results.sqlite")), blackboard.config['performance'].getint('search nodes') * 5)
    return
