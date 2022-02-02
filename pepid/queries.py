import numpy
import blackboard
from os import path
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

def fill_queries(start, end):
    """
    Processes batch of queries and put them in the temporary database for futher processing.
    """
    
    f = open(blackboard.config['data']['queries'], 'r')

    tol_l, tol_r = list(map(lambda x: float(x.strip()), blackboard.config['search']['candidate filtering tolerance'].split(",")))
    is_ppm = blackboard.config['search']['filtering unit'] == "ppm"

    min_mass = blackboard.config['database'].getfloat('min mass')
    max_mass = blackboard.config['database'].getfloat('max mass')
    min_charge = blackboard.config['search'].getint('min charge')
    max_charge = blackboard.config['search'].getint('max charge')

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
                if (min_mass <= precmass <= max_mass) and (min_charge <= charge <= max_charge):
                    data.append({k:None for k in blackboard.QUERY_COLS})
                    delta_l = tol_l if not is_ppm else pepid_utils.calc_rev_ppm(precmass, tol_l)
                    delta_r = tol_r if not is_ppm else pepid_utils.calc_rev_ppm(precmass, tol_r)
                    data[-1]['title'] = title
                    data[-1]['rt'] = rt
                    data[-1]['charge'] = charge
                    data[-1]['mass'] = precmass
                    data[-1]['spec'] = blackboard.Spectrum(list(zip(mz_arr, intens_arr)))
                    data[-1]['min_mass'] = precmass + delta_l
                    data[-1]['max_mass'] = precmass + delta_r
                    data[-1]['meta'] = blackboard.Meta(None)
                    
            elif '0' <= l[0] <= '9':
                mz, intens = l.split(maxsplit=1)
                mz = float(mz)
                if mz > 100:
                    mz_arr.append(mz)
                    intens_arr.append(float(intens))
    cur = blackboard.CONN.cursor()
    blackboard.executemany(cur, blackboard.insert_dict_str("queries", blackboard.QUERY_COLS), data)
    cur.close()
    blackboard.commit()

def user_processing(start, end):
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

    cur = blackboard.CONN.cursor()
    blackboard.execute(cur, blackboard.select_str("queries", blackboard.QUERY_COLS + ["rowid"], "WHERE rowid BETWEEN ? AND ?"), (start+1, end))
    data = cur.fetchall()
    meta = metadata_fn(data[:end-start])
    blackboard.executemany(cur, "UPDATE queries SET meta = ? WHERE rowid = ?;", zip(map(blackboard.Meta, meta), map(lambda x: x['rowid'], data)))
    cur.close()

def prepare_db():
    """
    Creates the required tables in the temporary database for queries processing
    """

    cur = blackboard.CONN.cursor()
    blackboard.execute(cur, "DROP TABLE IF EXISTS queries;")
    blackboard.execute(cur, blackboard.create_table_str("q.queries", blackboard.QUERY_COLS, blackboard.QUERY_TYPES))
    cur.close()
    blackboard.commit()
