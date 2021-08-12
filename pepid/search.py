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

def identipy_rnhs2(cands, query):
    import scipy
    from scipy import spatial
    ret = []
    scores = []

    spectrum = numpy.array(query['spec'])
    mz_array = spectrum[:,0]
    charge = query['charge']
    qmass = query['mass']

    nterm = blackboard.config['search'].getfloat('nterm cleavage')
    cterm = blackboard.config['search'].getfloat('cterm cleavage')
    tree = scipy.spatial.cKDTree(mz_array.reshape((mz_array.size, 1)))

    norm = spectrum[:,1].sum()

    acc_ppm = blackboard.config['search']['matching unit'] == 'ppm'
    acc = blackboard.config['search'].getfloat('peak matching tolerance')

    for i in range(cands[1]):
        cand = cands[0][i]
        theoretical = numpy.zeros((cand.npeaks,1)) # xxx: needs to be dict, etc.
        for j in range(cand.npeaks):
            theoretical[j][0] = cand.spec[j][0]

        seq = cand.sequence.decode('ascii')
        seq_mass = cand.mass

        score = 0
        mult = []
        match = {}
        match2 = {}
        total_matched = 0
        sumI = 0

        dist_all = []
        fragments = theoretical
        dist, ind = tree.query(fragments, distance_upper_bound=acc if not acc_ppm else (float(acc) / 1e6 * 2000))
        mask1 = (dist != numpy.inf)
        if acc_ppm:
            nacc = numpy.where(dist[mask1] / mz_array[ind[mask1]] * 1e6 > acc)[0]
            mask2 = mask1.copy()
            mask2[nacc] = False
        else:
            mask2 = mask1
        nmatched = mask2.sum()
        if nmatched:
            total_matched += nmatched
            mult.append(numpy.math.factorial(nmatched))
            sumi = spectrum[:,1][ind[mask2]].sum()
            sumI += sumi
            score += sumi / norm
            dist_all.extend(dist[mask2])
        match['!'] = mask2
        match2['!'] = mask2
        if not total_matched:
            ret.append({})
            scores.append(0)
            continue
        for m in mult:
            score *= m
        sumI = numpy.log10(sumI)

        ret.append({'score': score, 'theoretical': theoretical, 'spec': mz_array.tolist(), 'match2': {k:v.tolist() for k, v in match2.items()}, 'match': {k:v.tolist() for k, v in match.items()}, 'sumI': sumI, 'dist': dist_all, 'total_matched': total_matched})
        scores.append(score)
    return scores, ret

def identipy_rnhs(cands, query):
    import scipy
    from scipy import spatial
    ret = []
    scores = []

    spectrum = numpy.array(query['spec'])
    mz_array = spectrum[:,0]
    charge = query['charge']
    qmass = query['mass']

    nterm = blackboard.config['search'].getfloat('nterm cleavage')
    cterm = blackboard.config['search'].getfloat('cterm cleavage')
    tree = scipy.spatial.cKDTree(mz_array.reshape((mz_array.size, 1)))

    norm = spectrum[:,1].sum()

    acc_ppm = blackboard.config['search']['matching unit'] == 'ppm'
    acc = blackboard.config['search'].getfloat('peak matching tolerance')

    for i in range(cands[1]):
        cand = cands[0][i]
        theoretical = numpy.zeros((cand.npeaks,)) # xxx: needs to be dict, etc.
        for j in range(cand.npeaks):
            theoretical[j] = cand.spec[j][0]

        seq = cand.sequence.decode('ascii')
        seq_mass = cand.mass

        score = 0
        mult = []
        match = {}
        match2 = {}
        total_matched = 0
        sumI = 0

        dist_all = []
        for ion, fragments in theoretical.items():
            if ion[-1] >= charge:
                break
            dist, ind = tree.query(fragments, distance_upper_bound=acc if not acc_ppm else (float(acc) / 1e6 * 2000))
            mask1 = (dist != numpy.inf)
            if acc_ppm:
                nacc = numpy.where(dist[mask1] / mz_array[ind[mask1]] * 1e6 > acc)[0]
                mask2 = mask1.copy()
                mask2[nacc] = False
            else:
                mask2 = mask1
            nmatched = mask2.sum()
            if nmatched:
                total_matched += nmatched
                mult.append(numpy.math.factorial(nmatched))
                sumi = spectrum[:,1][ind[mask2]].sum()
                sumI += sumi
                score += sumi / norm
                dist_all.extend(dist[mask2])
            match[ion] = mask2
            match2[ion] = mask2
        if not total_matched:
            ret.append(None)
            scores.append(0)
            continue
        for m in mult:
            score *= m
        sumI = numpy.log10(sumI)

        ret.append({'score': score, 'theoretical': theoretical, 'spec': mz_array.tolist(), 'match2': {k:v.tolist() for k, v in match2.items()}, 'match': {k:v.tolist() for k, v in match.items()}, 'sumI': sumI, 'dist': dist_all, 'total_matched': total_matched})
        scores.append(score)
    return scores, ret

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

    spec = numpy.array(query['spec'])[:query['npeaks']]

    ret = []
    scores = []
    enzyme = re.compile(blackboard.config['database']['digestion'])
    varmods = set(map(lambda x: x.strip()[0], blackboard.config['search']['variable modifications'].split(",")))
    
    for ic in range(cands[1]):
        cand = cands[0][ic]
        th_masses = numpy.zeros((cand.npeaks,))
        for ipeak in range(cand.npeaks):
            th_masses[ipeak] = cand.spec[ipeak][0]
        th_masses = th_masses.reshape((-1,))
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
                    'varmods': float((numpy.array(mods) > 0).sum()) / max(1, sum([x in varmods for x in cand.sequence.decode('ascii')])),
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

    queries = numpy.memmap(os.path.join(blackboard.config['data']['tmpdir'], "query{}.npy".format(start)), dtype=blackboard.QUERY_DTYPE, shape=blackboard.config['performance'].getint('batch size'), mode='r')

    scoring_fn = rnhs
    try:
        mod, fn = blackboard.config['scoring']['score function'].rsplit('.', 1)
        user_fn = getattr(__import__(mod, fromlist=[fn]), fn)
        scoring_fn = user_fn
    except:
        import sys
        sys.stderr.write("[search]: user scoring function not found, using default scorer instead")

    batch_size = blackboard.config['performance'].getint('score batch size')

    import sys
    import time

    for iq, q in enumerate(queries):
        sys.stderr.write("[{}] Q {}/{}\n".format(time.strftime("%H:%M:%S", time.localtime()), iq, len(queries)))
        if(q['mass'] > 0):
            cands_start, cands_end = helper.find(blackboard.DB_PATH.rsplit(".bin", 1)[0] + "_index.bin", helper.KEY_TYPE, q['min_mass'], q['max_mass'])
            if cands_start >= 0:
                if cands_end < 0:
                    cands_end = cands_start
                for i in range(cands_start, cands_end, batch_size):
                    sys.stderr.write("[{}] B {}/{}\n".format(time.strftime("%H:%M:%S", time.localtime()), (i - cands_start) // batch_size, ((cands_end - cands_start + 1) // batch_size)))
                    cands, n_cands = helper.load_db(blackboard.DB_PATH, start=i, end=min(i + batch_size - 1, cands_end) + 1)
                    if(n_cands > 0):
                        scores, score_data = scoring_fn((cands, n_cands), q)
                        results = helper.make_res(scores, score_data, q, cands)
                        helper.dump_res(blackboard.DB_PATH.rsplit(".bin", 1)[0] + "_search.bin", results, len(scores), offset=start + i - cands_start)
                        helper.free(results)
                        helper.free(cands)

def prepare_search():
    """
    Creates necessary tables in the temporary database for final search results
    """

    pass
