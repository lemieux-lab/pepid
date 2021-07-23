import numpy
from os import path
import blackboard
import pepid_utils
import time
import random
import os
import re
import helper

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

    for cand in cands:
        theoretical = cand['spec']

        seq = cand['sequence']
        seq_mass = cand['mass']

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
    
    for cand in cands:
        th_masses = numpy.array(cand['spec'][:cand['npeaks']])[:,0]
        th_masses = th_masses.reshape((-1,))
        th_masses.sort()

        all_masses = numpy.repeat(th_masses, spec.shape[0]).reshape((-1, spec.shape[0])).T
        all_dists = numpy.abs(all_masses - spec[:,0].reshape((-1, 1)))

        if not is_ppm:
            deltas = [tol]*len(th_masses)
        else:
            deltas = [pepid_utils.calc_rev_ppm(m, tol) for m in th_masses]
        deltas = numpy.array(deltas).reshape((1, -1))
        dist_mask = (all_dists <= deltas)
        # For each ion series, how many peaks matched any theoretical peak
        n_matches = dist_mask.sum(axis=0)

        if n_matches.sum() == 0:
            scores.append(0)
            ret.append({})
            continue

        mult = numpy.math.factorial(int(n_matches.sum())) #numpy.array([numpy.math.factorial(n) for n in n_matches.astype('int32')]).prod()
        intens_sum = spec[:,1].sum()
        intens_score = sum([spec[dist_mask[:,i]][:,1].sum() / intens_sum for i in range(dist_mask.shape[1])])
        rnhs_score = intens_score * mult

        mascot_t = -10*numpy.log10(0.05 * (1.0 / len(cands)))-13
        mc = len(re.findall(enzyme, cand['sequence']))-1
        ret.append({"mScore": rnhs_score, "dM": cand['mass'] - query['mass'], "MIT": mascot_t, "MHT": mascot_t,
                    "peptideLength": len(cand['sequence']), "z1": int(query['charge'] == 1), "z2": int(2 <= query['charge'] <= 3),
                    "z4": int(4 <= query['charge'] <= 6), "z7": int(query['charge'] >= 7), "isoDM": abs(cand['mass'] - query['mass']),
                    "isoDMppm": abs(pepid_utils.calc_ppm(cand['mass'], query['mass'])), "isoDmz": abs(cand['mass'] - query['mass']),
                    "12C": 1, "mc0": int(mc == 0), "mc1": int(0 <= mc <= 1), "mc2": int(mc >= 2),
                    'varmods': float((numpy.array(cand['mods']) > 0).sum()) / max(1, sum([x in varmods for x in cand['sequence']])),
                    'varmodsCount': len(numpy.unique(cand['mods'])), 'totInt': numpy.log10(intens_sum),
                    'intMatchedTot': numpy.log10(sum([spec[dist_mask[:,i]][:,1].sum() for i in range(dist_mask.shape[1])])),
                    'relIntMatchedTot': intens_score, 'RMS': numpy.sqrt((all_dists[dist_mask]**2).mean()),
                    'RMSppm': numpy.sqrt((((all_dists[dist_mask] / all_masses[dist_mask]) * 1e6)**2).mean()),
                    'meanAbsFragDa': all_dists[dist_mask].mean(), 'meanAbsFragPPM': (all_dists[dist_mask] / all_masses[dist_mask]).mean(),
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
    results = numpy.memmap(os.path.join(blackboard.config['data']['tmpdir'], "results{}.npy".format(start)), dtype=blackboard.RES_DTYPE, shape=blackboard.config['performance'].getint('batch size'), mode='w+')

    scoring_fn = rnhs
    try:
        mod, fn = blackboard.config['scoring']['score function'].rsplit('.', 1)
        user_fn = getattr(__import__(mod, fromlist=[fn]), fn)
        scoring_fn = user_fn
    except:
        import sys
        sys.stderr.write("[search]: user scoring function not found, using default scorer instead")

    query_batch = {}
    cand_batch = {}

    import sys
    for q in queries:
        if(q['mass'] > 0):
            cands_start, cands_end = helper.find(blackboard.DB_PATH, helper.Db, q['min_mass'], q['max_mass'])
            if cands_start >= 0:
                if cands_end < 0:
                    cands_end = cands_start
                query_batch[q['title']] = q
                cand_batch[q['title']] = helper.load_db(blackboard.DB_PATH, start=cands_start, end=cands_end+1)

    for i, k in enumerate(query_batch.keys()):
        scores, score_data = scoring_fn(cand_batch[k], query_batch[k])

        for j in range(len(scores)):
            results[i]['title'] = k
            results[i]['description'] = cand_batch[k][j]['description']
            results[i]['seq'] = cand_batch[k][j]['sequence']
            results[i]['modseq'] = cand_batch[k][j]['mods']
            results[i]['length'] = len(cand_batch[k][j]['sequence'])
            results[i]['calc_mass'] = cand_batch[k][j]['mass']
            results[i]['mass'] = query_batch[k]['mass']
            results[i]['rt'] = query_batch[k]['rt']
            results[i]['charge'] = query_batch[k]['charge']
            results[i]['score'] = scores[j]
            results[i]['score_data'] = score_data[j]
        results.flush()

def prepare_search():
    """
    Creates necessary tables in the temporary database for final search results
    """

    pass
