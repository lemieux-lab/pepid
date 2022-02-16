import numpy
from os import path
import blackboard
import pepid_utils
import time
import random
import os
import re
import helper
import functools
import pickle
import db

def crnhs(cands, queries, whole_data=True):
    ppm = blackboard.config['search']['matching unit'] == 'ppm'
    tol = blackboard.config['search'].getfloat('peak matching tolerance')
    ret = helper.rnhs(queries, cands, tol, ppm, whole_data)
    return ret

def identipy_rnhs2(cands, q, whole_data=True):
    import scipy
    from scipy import spatial
    ret = []
    theoreticals = []

    for i in range(len(cands)):
        spectrum = numpy.asarray(q[i]['spec'].data[:blackboard.config['search'].getint('max peaks')])
        mz_array = spectrum[:,:1]
        intens = spectrum[:,1]
        norm = intens.sum()
        charge = q[i]['charge']
        qmass = q[i]['mass']

        nterm = blackboard.config['search'].getfloat('nterm cleavage')
        cterm = blackboard.config['search'].getfloat('cterm cleavage')
        #tree = scipy.spatial.cKDTree(mz_array)
        #import faiss
        #tree = faiss.IndexFlatL2(1)
        #tree.add(numpy.ascontiguousarray(mz_array))
        import sklearn
        import sklearn.neighbors
        #tree = sklearn.neighbors.KDTree(mz_array, leaf_size=16)

        acc_ppm = blackboard.config['search']['matching unit'] == 'ppm'
        acc = blackboard.config['search'].getfloat('peak matching tolerance')

        upper_bound = acc if not acc_ppm else (float(acc) / 1e6 * 2000)
        #theoretical = numpy.ndarray(buffer=cand.spec, dtype='float32', shape=(cand.npeaks,2))[:,:1] # xxx: needs to be dict, etc.
        query_peaks = blackboard.config['search'].getint('max peaks')
        c = cands[i]
        theoretical = c['spec'].data.astype('float32').reshape((-1, 1))
        theoreticals.append(theoretical)
        seqs = c['seq']
        seq_mass = c['mass']
        tree = sklearn.neighbors.KDTree(theoretical)
        dist_raw, ind = tree.query(mz_array, k=1, sort_results=False, dualtree=False)

        #theoretical = numpy.ndarray(buffer=cand.spec, dtype='float32', shape=(cand.npeaks,2))[:,:1] # xxx: needs to be dict, etc.
        #theoretical = numpy.ascontiguousarray(numpy.ndarray(buffer=cand.spec, dtype='float32', shape=(cand.npeaks,2))[:,:1]) # xxx: needs to be dict, etc.

        score = 0
        mult = []
        total_matched = 0
        sumI = 0

        ind = ind[dist_raw < upper_bound]
        dist = dist_raw[dist_raw < upper_bound]
        nmatched = len(ind)
        if nmatched > 0:
            total_matched += nmatched
            mult.append(numpy.math.factorial(nmatched))
            sumi = sum(intens[dist_raw[:,0] < upper_bound])
            sumI += sumi
            score += sumi / norm
        if not total_matched:
            ret.append({'score': 0, 'theoretical': None, 'spec': None, 'sumI': 0, 'dist': None, 'total_matched': 0, 'title': q[i]['title'], 'desc': c['desc'], 'seq': None, 'modseq': None})
            continue
        for m in mult:
            score *= m
        if sumI != 0:
            sumI = numpy.log10(sumI)

        ret.append({'score': score, 'theoretical': theoreticals[i].tolist(), 'spec': mz_array.tolist(), 'sumI': sumI, 'dist': dist.tolist(), 'total_matched': total_matched, 'title': q[i]['title'], 'desc': c['desc'], 'seq': c['seq'], 'modseq': "".join([s if m == 0 else s + "[{}]".format(m) for s,m in zip(c['seq'], c['mods'])])})
    return ret

def binary_search(q, lst):
    low = 0
    high = len(lst) - 1
    ptr = 0

    best_idx = ptr
    best = 9999

    while high >= low:
        ptr = (high + low) // 2
        if lst[ptr][0] > q:
            high = ptr - 1
            dist = abs(lst[ptr][0] - q)
            if dist < best:
                best = dist
                best_idx = ptr
        elif lst[ptr][0] < q:
            low = ptr + 1
            dist = abs(lst[ptr][0] - q)
            if dist < best:
                best = dist
                best_idx = ptr
        else:
            best_idx = ptr
            break
    return best_idx
    

class FakeTree(object):
    def __init__(self, data):
        self.data = data

    def query_bin0(self, queries, distance_upper_bound):
        dist = [float('inf')] * len(queries)
        indices = [-1] * len(queries)

        prev_ind = binary_search(queries[0][0], self.data)

        for iq, q in enumerate(queries):
            for i in range(prev_ind, len(self.data)):
                d = self.data[i][0] - q[0]
                absd = abs(d)
                if absd <= distance_upper_bound:
                    if absd < dist[iq]:
                        dist[iq] = absd
                        indices[iq] = i
                elif d < 0:
                    prev_ind = i
                else:
                    break
        return numpy.array(dist), numpy.array(indices)

    def query_bin(self, queries, distance_upper_bound):
        dist = [float('inf')] * len(queries)
        indices = [-1] * len(queries)

        i = 0

        start = binary_search(self.data[0][0], queries)
        end = binary_search(self.data[-1][0], queries)
        for iq in range(start, end+1):
            i += binary_search(queries[iq][0], self.data[i:])
            d = self.data[i][0] - queries[iq][0]
            absd = abs(d)
            if absd <= distance_upper_bound:
                if absd < dist[iq]:
                    dist[iq] = absd
                    indices[iq] = i
            iq += 1
        return numpy.array(dist), numpy.array(indices)

    def query_tree(self, queries, distance_upper_bound):
        import scipy
        from scipy import spatial
        tree = scipy.spatial.cKDTree(self.data)
        return tree.query(queries, distance_upper_bound=distance_upper_bound)

    def query_npy(self, queries, distance_upper_bound):
        qblock = numpy.repeat(queries, len(self.data), axis=1)
        cblock = numpy.repeat(self.data.T, len(queries), axis=0)

        dblock = numpy.abs(qblock - cblock)
        ind = dblock.argmin(axis=1)
        dist = dblock.min(axis=1)
        mask = dist > distance_upper_bound
        ind[mask] = -1
        dist[mask] = numpy.inf
        
        return dist, ind

    def query(self, queries, distance_upper_bound):
        return self.query_npy(queries, distance_upper_bound)

def identipy_rnhs(cands, q, whole_data=True):
    import scipy
    from scipy import spatial
    ret = []
    scores = []

    for i in range(len(cands)):
        spectrum = numpy.asarray(q[i]['spec'].data)
        mz_array = spectrum[:,0]
        intens = spectrum[:,1]
        norm = intens.sum()
        charge = q[i]['charge']
        qmass = q[i]['mass']

        nterm = blackboard.config['search'].getfloat('nterm cleavage')
        cterm = blackboard.config['search'].getfloat('cterm cleavage')

        acc_ppm = blackboard.config['search']['matching unit'] == 'ppm'
        acc = blackboard.config['search'].getfloat('peak matching tolerance')

        query_peaks = blackboard.config['search'].getint('max peaks')
        c = cands[i]
        theoretical = c['spec'].data
        seqs = c['seq']
        seq_mass = c['mass']

        score = 0
        mult = []
        match = {}
        match2 = {}
        total_matched = 0
        sumI = 0

        for ifrag, fragments in enumerate(theoretical):
            qblock = numpy.repeat(mz_array.reshape((-1, 1)), len(fragments), axis=1)

            sumi = 0
            nmatched = 0

            if (ifrag // 2 + 1) >= charge:
                break

            cblock = numpy.repeat(fragments.T, len(mz_array), axis=0)

            dblock = numpy.abs(qblock - cblock)

            dist = dblock.min(axis=1)

            mask = (dist <= acc) if not acc_ppm else (dist / mz_array * 1e6 <= acc)
            sumi += intens[mask].sum()
            nmatched += mask.sum()

            if nmatched > 0:
                total_matched += nmatched
                mult.append(numpy.math.factorial(nmatched))
                sumI += sumi
                score += sumi / norm

        if total_matched == 0:
            ret.append({'score': 0, 'theoretical': None, 'spec': None, 'sumI': 0, 'dist': None, 'total_matched': 0, 'title': q[i]['title'], 'desc': c['desc'], 'seq': None, 'modseq': None})
            continue

        for m in mult:
            score *= m
        sumI = numpy.log10(sumI)

        ret.append({'score': score, 'theoretical': theoretical, 'spec': mz_array.tolist(), 'sumI': sumI, 'dist': dist.tolist(), 'total_matched': total_matched, 'title': q[i]['title'], 'desc': c['desc'], 'seq': c['seq'], 'modseq': "".join([s if m == 0 else s + "[{}]".format(m) for s,m in zip(c['seq'], c['mods'])])})
    return ret

def rnhs(cands, query):
    """
    Example scoring function: rnhs from identipy.
    Score output is a pair: score value and a dictionary. If the score function will be rescored by mascot,
    the dictionary keys corresponding to the mascot parameters will be used for that purpose.    

    Here, data is input based on the percolator documentation as well as
    http://www.matrixscience.com/help/interpretation_help.html
    (e.g. for the -13 empirical correction of MIT)
    """

    nterm = blackboard.config['search'].getfloat('nterm cleavage')
    cterm = blackboard.config['search'].getfloat('cterm cleavage')

    is_ppm = blackboard.config['search']['matching unit'] == 'ppm'
    tol = blackboard.config['search'].getfloat('peak matching tolerance')

    spec = numpy.ndarray(buffer=query['spec'], dtype='float32', shape=(query['npeaks'], 2))

    ret = []
    scores = []
    enzyme = re.compile(blackboard.config['database']['digestion'])
    varmods = set(map(lambda x: x.strip()[0], blackboard.config['search']['variable modifications'].split(",")))
    
    for ic in range(cands[1]):
        cand = cands[0][ic]
        th_masses = numpy.zeros((cand.npeaks,))
        for ipeak in range(cand.npeaks):
            th_masses[ipeak] = cand.spec[ipeak][0]
        th_masses.sort()

        #all_masses = numpy.repeat(th_masses, spec.shape[0]).reshape((-1, spec.shape[0])).T
        # numpy.repeat seems to be broken (hangs indefinitely), do it manually instead
        all_dists = numpy.zeros((spec.shape[0], th_masses.shape[0]))
        for i in range(spec.shape[0]):
            for j in range(th_masses.shape[0]):
                all_dists[i,j] = numpy.abs(th_masses[j] - spec[i,0])

        if not is_ppm:
            deltas = [tol]*len(th_masses)
        else:
            deltas = th_masses * 1e-6 * tol
        dist_mask = (all_dists <= deltas)
        # For each ion series, how many peaks matched any theoretical peak
        #n_matches = dist_mask.sum(axis=0)
        # numpy sum is broken, so do it manually
        n_matches = functools.reduce(lambda x, y: x + y, dist_mask)

        if n_matches.sum() == 0:
            scores.append(0)
            ret.append({})
            continue

        mult = numpy.math.factorial(int(n_matches.sum()))
        intens_sum = spec[:,1].sum()
        intens_score = sum([spec[dist_mask[:,i]][:,1].sum() / intens_sum for i in range(dist_mask.shape[1])])
        rnhs_score = intens_score * mult

        mods = numpy.zeros((cand.length,))
        for im in range(cand.length):
            mods[im] = cand.mods[im]

        mascot_t = -10*numpy.log10(0.05 * (1.0 / cands[1]))-13
        mc = len(re.findall(enzyme, cand.sequence.decode('ascii')))-1
        #ret.append({"mScore": 0})
        ret.append({"mScore": rnhs_score, "dM": cand.mass - query['mass'], "MIT": mascot_t, "MHT": mascot_t,
                    "peptideLength": len(cand.sequence.decode('ascii')), "z1": int(query['charge'] == 1), "z2": int(2 <= query['charge'] <= 3),
                    "z4": int(4 <= query['charge'] <= 6), "z7": int(query['charge'] >= 7), "isoDM": abs(cand.mass - query['mass']),
                    "isoDMppm": abs(pepid_utils.calc_ppm(cand.mass, query['mass'])), "isoDmz": abs(cand.mass - query['mass']),
                    "12C": 1, "mc0": int(mc == 0), "mc1": int(0 <= mc <= 1), "mc2": int(mc >= 2),
                    'varmods': float((numpy.asarray(mods) > 0).sum()) / max(1, sum([x in varmods for x in cand.sequence.decode('ascii')])),
                    'varmodsCount': len(numpy.unique(mods)), 'totInt': numpy.log10(intens_sum),
                    'intMatchedTot': numpy.log10(sum([spec[dist_mask[:,i]][:,1].sum() for i in range(dist_mask.shape[1])])),
                    'relIntMatchedTot': intens_score, 'RMS': numpy.sqrt((all_dists[dist_mask]**2).mean()),
                    #'RMSppm': numpy.sqrt((((all_dists[dist_mask] / all_masses[dist_mask]) * 1e6)**2).mean()),
                    'meanAbsFragDa': all_dists[dist_mask].mean(), #'meanAbsFragPPM': (all_dists[dist_mask] / all_masses[dist_mask]).mean(),
                    'rawscore': intens_score})
        scores.append(rnhs_score)
    return scores, ret

def search_core(start, end):
    """
    Core search algorithm: collects and finalizes the data, then
    parses and applies the user scoring function from the config.
    The function should take two arguments: a list of candidates, and a query.

    A query is a dictionary with the following keys:
    title: query title
    rt: retention time in seconds
    charge: precursor charge
    mass: precursor mass
    meta: user metadata (see fill_queries in queries.py)
    spec: N x 2 list representing spectrum ([[mz, intensity], ...])

    Each candidate is a dictionary with the following keys:
    description: sequence description from the database for the protein from which the peptide was generated
    sequence: the peptide sequence
    mods: an array of floating point numbers containing mass offsets for each amino acid
    rt: retention time in seconds
    length: length of the peptide sequence
    mass: calculated sequence mass
    spec: N x 2 list representing predicted sequence spectrum ([[mz, intensity], ...])
    """

    tol = blackboard.config['search'].getfloat('peak matching tolerance')
    is_ppm = blackboard.config['search']['matching unit'] == "ppm"

    min_mass = blackboard.config['database'].getfloat('min mass')
    max_mass = blackboard.config['database'].getfloat('max mass')
    max_peaks = blackboard.config['search'].getint('max peaks')

    scoring_fn = crnhs
    try:
        mod, fn = blackboard.config['scoring']['score function'].rsplit('.', 1)
        user_fn = getattr(__import__(mod, fromlist=[fn]), fn)
        scoring_fn = user_fn
    except:
        import sys
        sys.stderr.write("[search]: user scoring function not found, using default scorer instead")

    batch_size = blackboard.config['performance'].getint('score batch size')

    cur = blackboard.CONN.cursor()
    res_cur = blackboard.RES_CONN.cursor()

    blackboard.execute(cur, blackboard.select_str("queries", ["rowid"] + blackboard.QUERY_COLS, "WHERE rowid BETWEEN ? AND ?"), (start+1, end))
    queries = cur.fetchall()

    import sys
    for iq, q in enumerate(queries):
        blackboard.execute(cur, blackboard.select_str("candidates", ["rowid"] + blackboard.DB_COLS, "WHERE mass BETWEEN ? AND ?"), (q['min_mass'], q['max_mass']))
        
        while True:
            cands = cur.fetchmany(batch_size)
            if len(cands) == 0:
                break

            all_q = [q] * len(cands)

            res = scoring_fn(cands, all_q, whole_data=True)
            #sys.stderr.write("GOT SCORE!!\n")
            #import sys
            r = [{'data': b'', 'candrow': c['rowid'], 'qrow': q['rowid'], 'score': r['score'], 'title': r['title'], 'desc': r['desc'], 'modseq': r['modseq'], 'seq': r['seq']} for r, c in zip(res, cands)]
            blackboard.executemany(res_cur, blackboard.maybe_insert_dict_str("results", blackboard.RES_COLS), r)
            blackboard.RES_CONN.commit()
    blackboard.execute(res_cur, "CREATE INDEX IF NOT EXISTS res_score_qrow_idx ON results (qrow ASC, score DESC);")
    blackboard.RES_CONN.commit()

def prepare_search():
    """
    Creates necessary tables in the temporary database for final search results
    """

    pass
