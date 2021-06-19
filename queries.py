import numpy
import blackboard
from os import path
import psycopg2
import random
import pepid_utils
import time
import random
import os

def count_queries():
    """
    Opens the file speficied in config and counts how many spectra are present.
    """
    f = open(blackboard.config['data']['queries'], 'r')
    cnt = 0
    for l in f:
        cnt += l[:len("BEGIN IONS")] == "BEGIN IONS"
    return cnt

def fill_queries(conn, start, end):
    """
    Processes batch of queries and put them in the temporary database for futher processing.
    """
    cursor = conn.cursor()
    f = open(blackboard.config['data']['queries'], 'r')

    tol_l, tol_r = list(map(lambda x: float(x.strip()), blackboard.config['search']['candidate filtering tolerance'].split(",")))
    is_ppm = blackboard.config['search']['filtering unit'] == "ppm"

    data = []
    entry_idx = -1
    for l in f:
        if l[:len("BEGIN IONS")] == "BEGIN IONS":
            entry_idx += 1
            title = ""
            charge = 0
            precmass = 0.0
            rt = 0.0
            mz_arr = []
            intens_arr = []
        if entry_idx < start:
            continue
        elif entry_idx >= end:
            break
        else:
            if l[:len("TITLE=")] == "TITLE=":
                title = l[len("TITLE="):].strip()
            elif l[:len("RTINSECONDS=")] == "RTINSECONDS=":
                rt = int(l[len("RTINSECONDS="):].strip())
            elif l[:len("CHARGE=")] == "CHARGE=":
                charge = int(l[len("CHARGE="):].strip().replace("+", ""))
            elif l[:len("PEPMASS=")] == "PEPMASS=":
                precmass = float(l[len("PEPMASS="):].split(maxsplit=1)[0])
            elif l[:len("END IONS")] == "END IONS":
                precmass = (precmass * charge) - charge
                data.append([title, rt, charge, precmass, list(zip(mz_arr, intens_arr)), None])
            elif '0' <= l[0] <= '9':
                mz, intens = l.split(maxsplit=1)
                mz_arr.append(float(mz))
                intens_arr.append(float(intens))

    for (title, rt, charge, precmass, spec, meta) in data:
        delta_l = tol_l if not is_ppm else pepid_utils.calc_rev_ppm(precmass, tol_l)
        delta_r = tol_r if not is_ppm else pepid_utils.calc_rev_ppm(precmass, tol_r)
        done = False
        while not done:
            try:
                cursor.execute("INSERT INTO queries (title, rt, charge, mass, min_mass, max_mass, meta, spec) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);", [title, rt, charge, precmass, precmass + delta_l, precmass + delta_r, repr(None), repr(spec)])
                conn.commit()
                done = True
            except psycopg2.errors.DeadlockDetected:
                time.sleep(random.random())
                continue

def user_processing(conn, start, end):
    """
    Resolves the user-specified queries post-processing function from the config file,
    then applies it on a batch of processed queries from the database.

    The user function should accept a batch of queries and return a batch of metadata, which are
    inserted in the database for the corresponding input query.

    The keys for the query dictionary received by the user function are:
    title: query title
    rt: retention time in seconds
    charge: precursor charge
    mass: precursor neutral mass
    spec: a N x 2 list representing the spectrum ([[mz, intensity] ...])

    The output metadata object should be anything such that eval(expr(metadata)) == metadata, where == is defined
    in the sense of the user scoring function (metadata is not otherwise consulted).
    """
    
    metadata_fn = None
    try:
        mod, fn = blackboard.config['queries']['user processing function'].rsplit('.', 1)
        user_fn = getattr(__import__(mod, fromlist=[fn]), fn)
        metadata_fn = user_fn
    except:
        import sys
        sys.stderr.write("[queries post]: user processing function not found, not using extra processing\n")

    if metadata_fn is None:
        return

    cursor = conn.cursor()
    cursor.execute("SELECT title, rt, charge, mass, spec FROM queries WHERE rowid BETWEEN %s AND %s;", [start+1, end+1])
    data = list(cursor.fetchall())
    meta = metadata_fn([{"title": d[0], "rt": d[1], "charge": d[2], "mass": d[3], "spec": eval(d[4])} for d in data])
    for i, m in enumerate(meta):
        done = False
        while not done:
            try:
                cursor.execute("UPDATE queries SET meta = %s WHERE rowid = %s;", [repr(m), start+i+1])
                conn.commit()
                done = True
            except psycopg2.errors.DeadlockDetected:
                time.sleep(random.random())
                continue
            except psycopg2.DatabaseError as e:
                raise e

def prepare_queries():
    """
    Creates the required tables in the temporary database for queries processing
    """

    conn = None

    while conn is None:
        try:
            conn = psycopg2.connect(host="localhost", port='9991', database="postgres")
        except psycopg2.errors.DeadlockDetected:
            continue
        except psycopg2.DatabaseError as e:
            raise e

    os.environ['TMPDIR'] = blackboard.config['data']['tmpdir']
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE queries (rowid SERIAL, title TEXT PRIMARY KEY, rt REAL, charge INTEGER, mass REAL, min_mass REAL, max_mass REAL, meta TEXT, spec TEXT);")
    cursor.execute("CREATE INDEX q_idx on queries(mass, rt);")

    conn.commit()
    conn.close()
