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
    
    arr = numpy.memmap(os.path.join(blackboard.config['data']['tmpdir'], "query{}.npy".format(start)), mode="w+", dtype=blackboard.QUERY_DTYPE, shape=blackboard.config['performance'].getint('batch size'))
    f = open(blackboard.config['data']['queries'], 'r')

    tol_l, tol_r = list(map(lambda x: float(x.strip()), blackboard.config['search']['candidate filtering tolerance'].split(",")))
    is_ppm = blackboard.config['search']['filtering unit'] == "ppm"

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
                delta_l = tol_l if not is_ppm else pepid_utils.calc_rev_ppm(precmass, tol_l)
                delta_r = tol_r if not is_ppm else pepid_utils.calc_rev_ppm(precmass, tol_r)
                arr[entry_idx - start]['title'] = title
                arr[entry_idx - start]['rt'] = rt
                arr[entry_idx - start]['charge'] = charge
                arr[entry_idx - start]['mass'] = precmass
                max_peaks = blackboard.config['search'].getint('max peaks')
                arr[entry_idx - start]['spec'] = numpy.pad(numpy.array(list(zip(mz_arr, intens_arr)))[:max_peaks, :], ((0, max(0, max_peaks - len(mz_arr))), (0, 0)))
                arr[entry_idx - start]['npeaks'] = len(mz_arr)
                arr[entry_idx - start]['min_mass'] = precmass + delta_l
                arr[entry_idx - start]['max_mass'] = precmass + delta_r
                arr[entry_idx - start]['meta'] = repr(None)
                
            elif '0' <= l[0] <= '9':
                mz, intens = l.split(maxsplit=1)
                mz_arr.append(float(mz))
                intens_arr.append(float(intens))
    arr.flush()

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

    data = numpy.memmap(os.path.join(blackboard.config['data']['tmpdir'], "query{}.npy".format(start)), dtype=blackboard.QUERY_DTYPE, shape=blackboard.config['performance'].getint('batch size'), mode='r+')
    meta = metadata_fn(data[:end-start])
    data[:end-start]['meta'] = list(map(repr, meta))
    data.flush()

def prepare_queries():
    """
    Creates the required tables in the temporary database for queries processing
    """

    pass
