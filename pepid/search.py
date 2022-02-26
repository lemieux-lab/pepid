import numpy
import blackboard
import pepid_utils

def identipy_rnhs(cands, q):
    """
    Example scoring function: rnhs from identipy.
    Score output is a dictionary. If the score function will be rescored by mascot,
    the dictionary keys corresponding to the mascot parameters will be used for that purpose.    

    Here, data is output based on the percolator documentation as well as
    http://www.matrixscience.com/help/interpretation_help.html
    (e.g. for the -13 empirical correction of MIT)
    """

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

        acc_ppm = blackboard.config['scoring']['matching unit'] == 'ppm'
        acc = blackboard.config['scoring'].getfloat('peak matching tolerance')

        c = cands[i]
        theoretical = c['spec'].data
        seqs = c['seq']
        seq_mass = c['mass']

        score = 0
        mult = []
        total_matched = 0
        sumI = 0

        for ifrag, fragments in enumerate(theoretical):
            sumi = 0
            nmatched = 0

            if (ifrag // 2 + 1) >= charge:
                break

            qblock = numpy.repeat(mz_array.reshape((-1, 1)), len(fragments), axis=1)
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
            ret.append({'score': 0, 'theoretical': None, 'spec': None, 'sumI': 0, 'dist': None, 'total_matched': 0, 'title': q[i]['title'], 'desc': None, 'seq': None, 'modseq': None})
            continue

        for m in mult:
            score *= m
        sumI = numpy.log10(sumI)

        ret.append({'score': score, 'theoretical': theoretical, 'spec': mz_array.tolist(), 'sumI': sumI, 'dist': dist.tolist(), 'total_matched': total_matched, 'title': q[i]['title'], 'desc': c['desc'], 'seq': c['seq'], 'modseq': "".join([s if m == 0 else s + "[{}]".format(m) for s,m in zip(c['seq'], c['mods'])])})
        #ret.append({"mScore": rnhs_score, "dM": cand.mass - query['mass'], "MIT": mascot_t, "MHT": mascot_t,
        #            "peptideLength": len(cand.sequence.decode('ascii')), "z1": int(query['charge'] == 1), "z2": int(2 <= query['charge'] <= 3),
        #            "z4": int(4 <= query['charge'] <= 6), "z7": int(query['charge'] >= 7), "isoDM": abs(cand.mass - query['mass']),
        #            "isoDMppm": abs(pepid_utils.calc_ppm(cand.mass, query['mass'])), "isoDmz": abs(cand.mass - query['mass']),
        #            "12C": 1, "mc0": int(mc == 0), "mc1": int(0 <= mc <= 1), "mc2": int(mc >= 2),
        #            'varmods': float((numpy.asarray(mods) > 0).sum()) / max(1, sum([x in varmods for x in cand.sequence.decode('ascii')])),
        #            'varmodsCount': len(numpy.unique(mods)), 'totInt': numpy.log10(intens_sum),
        #            'intMatchedTot': numpy.log10(sum([spec[dist_mask[:,i]][:,1].sum() for i in range(dist_mask.shape[1])])),
        #            'relIntMatchedTot': intens_score, 'RMS': numpy.sqrt((all_dists[dist_mask]**2).mean()),
        #            #'RMSppm': numpy.sqrt((((all_dists[dist_mask] / all_masses[dist_mask]) * 1e6)**2).mean()),
        #            'meanAbsFragDa': all_dists[dist_mask].mean(), #'meanAbsFragPPM': (all_dists[dist_mask] / all_masses[dist_mask]).mean(),
        #            'rawscore': intens_score})

    return ret

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

    tol = blackboard.config['scoring'].getfloat('peak matching tolerance')
    is_ppm = blackboard.config['scoring']['matching unit'] == "ppm"

    scoring_fn = pepid_utils.import_or(blackboard.config['scoring']['function'], identipy_rnhs)

    batch_size = blackboard.config['scoring'].getint('batch size')

    cur = blackboard.CONN.cursor()
    res_cur = blackboard.RES_CONN.cursor()

    blackboard.execute(cur, blackboard.select_str("queries", ["rowid"] + blackboard.QUERY_COLS, "WHERE rowid BETWEEN ? AND ?"), (start+1, end))
    queries = cur.fetchall()

    for iq, q in enumerate(queries):
        blackboard.execute(cur, blackboard.select_str("candidates", ["rowid"] + blackboard.DB_COLS, "WHERE mass BETWEEN ? AND ?"), (q['min_mass'], q['max_mass']))
        
        while True:
            cands = cur.fetchmany(batch_size)
            if len(cands) == 0:
                break

            descs = []

            all_q = [q] * len(cands)

            res = scoring_fn(cands, all_q)
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
