import numpy
from os import path
import random
import time
import random
import os
import pickle

if __package__ is None or __package__ == '':
    import blackboard
    import pepid_utils
else:
    from . import blackboard
    from . import pepid_utils

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

    tol_l, tol_r = list(map(lambda x: float(x.strip()), blackboard.config['scoring']['candidate filtering tolerance'].split(",")))
    is_ppm = blackboard.config['scoring']['filtering unit'] == "ppm"

    min_mass = blackboard.config['processing.query'].getfloat('min mass')
    max_mass = blackboard.config['processing.query'].getfloat('max mass')
    min_charge = blackboard.config['processing.query'].getint('min charge')
    max_charge = blackboard.config['processing.query'].getint('max charge')

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
            continue

        if entry_idx < start:
            continue
        elif entry_idx >= end:
            break

        if l[:len("TITLE=")] == "TITLE=":
            title = l[len("TITLE="):].strip()
        elif l[:len("RTINSECONDS=")] == "RTINSECONDS=":
            rt = float(l[len("RTINSECONDS="):].strip())
        elif l[:len("CHARGE=")] == "CHARGE=":
            charge = int(l[len("CHARGE="):].strip().replace("+", ""))
        elif l[:len("PEPMASS=")] == "PEPMASS=":
            precmass = float(l[len("PEPMASS="):].split(maxsplit=1)[0])
        elif l[:len("END IONS")] == "END IONS":
            precmass = (precmass * charge) - (charge-1)*pepid_utils.MASS_PROT - pepid_utils.MASS_PROT
            if (min_mass <= precmass <= max_mass) and (min_charge <= charge <= max_charge):
                data.append({k:None for k in blackboard.QUERY_COLS})
                delta_l = tol_l if not is_ppm else pepid_utils.calc_rev_ppm(precmass, tol_l)
                delta_r = tol_r if not is_ppm else pepid_utils.calc_rev_ppm(precmass, tol_r)
                data[-1]['title'] = title
                data[-1]['rt'] = rt
                data[-1]['charge'] = charge
                data[-1]['mass'] = precmass
                data[-1]['spec'] = pickle.dumps(list(zip(mz_arr, intens_arr)))
                data[-1]['min_mass'] = precmass + delta_l
                data[-1]['max_mass'] = precmass + delta_r
                data[-1]['meta'] = pickle.dumps(None)
        elif '0' <= l[0] <= '9':
            mz, intens = l.split(maxsplit=1)
            mz_arr.append(float(mz))
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

    The query object is a database row, which functions similarly to a database and whose
    keys are as defined in `blackboard.py`

    The output metadata object should be anything such that eval(expr(metadata)) == metadata, where == is defined
    in the sense of the user scoring function (metadata is not otherwise consulted).
    """
    
    metadata_fn = pepid_utils.import_or(blackboard.config['processing.query']['postprocessing function'], None)

    if metadata_fn is None:
        return

    cur = blackboard.CONN.cursor()

    rows = set()
    rows.add('rowid')
    rows.update(getattr(metadata_fn, 'required_fields', {}).get('queries', []))

    blackboard.execute(cur, "SELECT {} FROM queries WHERE rowid BETWEEN ? AND ?;".format(",".join(rows)), (start+1, end))

    data = cur.fetchall()
    meta = metadata_fn(data)
    if meta is not None:
        blackboard.executemany(cur, "UPDATE queries SET meta = ? WHERE rowid = ?;", zip(map(pickle.dumps, meta), map(lambda x: x['rowid'], data)))
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
