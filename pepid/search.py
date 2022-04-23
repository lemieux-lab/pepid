import numpy
import numpy.linalg
import blackboard
import pepid_utils
import re
import scipy
import scipy.signal

def cosine(cands, q):
    """
    Simple cosine distance between two spectra.
    Requires both spectra to be simple (mz, intensity) lists.
    """

    blit_q = numpy.zeros((len(cands), 20000)) 

    for i in range(len(cands)):
        for mz, intens in q[i]['spec'].data:
            if mz >= 2000-0.5 or mz == 0:
                break
            blit_q[i, round(mz * 10)] = intens

    blit_cand = numpy.vstack([numpy.asarray(cands[i]['spec'].data[q[i]['charge']-1].todense()) for i in range(len(cands))])
    score = ((blit_q * blit_cand) / numpy.maximum(1e-5, (numpy.linalg.norm(blit_q, axis=-1, keepdims=True) * numpy.linalg.norm(blit_cand, axis=-1, keepdims=True)))).sum(axis=-1).reshape((-1,))

    ret = [{'score': score[i], 'theoretical': q[i]['spec'].data, 'spec': cands[i]['spec'].data[q[i]['charge']-1], 'sumI': 0, 'dist': None, 'total_matched': 0, 'title': q[i]['title'], 'desc': cands[i]['desc'], 'seq': cands[i]['seq'], 'modseq': "".join([s if m == 0 else s + "[{}]".format(m) for s,m in zip(cands[i]['seq'], cands[i]['mods'])])} for i in range(len(cands))]
    return ret

def identipy_rnhs(cands, q):
    """
    Sample scoring function: rnhs from identipy.
    Score output is a dictionary. If the score function will be rescored by mascot,
    the dictionary keys corresponding to the mascot parameters will be used for that purpose.    
    """

    ret = []
    scores = []

    style = blackboard.config['rnhs']['feature style']

    #if style != 'mascot':
    #    max_mz = 5000
    #    bin_mult = 50

    #    spec_blit = numpy.zeros((max_mz * bin_mult,))

    #    for im, iv in numpy.asarray(q[0]['spec'].data): # XXX hack since we know we always receive the same q...
    #        if im >= max_mz:
    #            break
    #        spec_blit[int(im * bin_mult)] = iv
    #    spec_blit = numpy.sqrt(spec_blit)
    #    spec_blit = spec_blit / spec_blit.max()

    for i in range(len(cands)):
        spectrum = numpy.asarray(q[i]['spec'].data)
        mz_array = spectrum[:,0]
        intens = spectrum[:,1]
        norm = intens.sum()
        charge = q[i]['charge']
        qmass = q[i]['mass']
        c = cands[i]

        #if style != 'mascot':
        #    xcorr = 0
        #    th_blit = numpy.zeros((max_mz * bin_mult,))

        #    for sub_spec in c['spec'].data:
        #        for im in sub_spec:
        #            if im >= max_mz:
        #                break
        #            th_blit[int(im * bin_mult)] = 1

        #    # Required to avoid passing over RLIMIT_NPROC if using sufficiently many parallel processes
        #    xcorr = (scipy.signal.correlate(th_blit, spec_blit, mode='full')[len(th_blit) - 75 * bin_mult : len(th_blit) + 75 * bin_mult]).sum() / 150

        acc_ppm = blackboard.config['scoring']['matching unit'] == 'ppm'
        acc = blackboard.config['scoring'].getfloat('peak matching tolerance')

        theoretical = c['spec'].data
        seqs = c['seq']
        seq_mass = c['mass']

        score = 0
        mult = []
        total_matched = 0
        sumI = 0
        masks = []
        dists = []

        for ifrag, fragments in enumerate(theoretical):
            sumi = 0
            nmatched = 0

            if (ifrag // 2 + 1) >= charge:
                break

            qblock = numpy.repeat(mz_array.reshape((-1, 1)), len(fragments), axis=1)
            cblock = numpy.repeat(fragments.T, len(mz_array), axis=0)
            dblock = numpy.abs(qblock - cblock)
            dist = dblock.min(axis=1)
            dists.append(dist)

            mask = (dist <= acc) if not acc_ppm else (dist / mz_array * 1e6 <= acc)
            masks.append(mask)
            sumi += intens[mask].sum()
            nmatched += mask.sum()

            if nmatched > 0:
                total_matched += nmatched
                mult.append(numpy.math.factorial(nmatched))
                sumI += sumi
                score += sumi / norm

        if total_matched == 0:
            ret.append({'score': 0})
            continue

        for m in mult:
            score *= m
        logsumI = numpy.log10(sumI)

        #ret.append({'score': score, 'theoretical': theoretical, 'spec': mz_array.tolist(), 'sumI': logsumI, 'dist': dist.tolist(), 'total_matched': total_matched, 'title': q[i]['title'], 'desc': c['desc'], 'seq': c['seq'], 'modseq': "".join([s if m == 0 else s + "[{}]".format(m) for s,m in zip(c['seq'], c['mods'])])})

        if style == 'mascot':
            ret.append({"dM": c['mass'] - q[i]['mass'], #"MIT": mascot_t, "MHT": mascot_t, "mScore": mascot
                        "peptideLength": len(c['seq']), "z1": int(q[i]['charge'] == 1), "z2": int(2 <= q[i]['charge'] <= 3),
                        "z4": int(4 <= q[i]['charge'] <= 6), "z7": int(q[i]['charge'] >= 7), "isoDM": abs(c['mass'] - q[i]['mass']),
            #            "isoDMppm": abs(pepid_utils.calc_ppm(c['mass'], q[i]['mass'])), "isoDmz": abs(c['mass'] - q[i]['mass']),
            #            "12C": 1, "mc0": int(mc == 0), "mc1": int(0 <= mc <= 1), "mc2": int(mc >= 2),
                        'varmods': float((numpy.asarray(c['mods']) > 0).sum()) / max(1, sum([x in c['mods'] for x in c['seq']])),
                        'varmodsCount': len(numpy.unique(c['mods'])), 'totInt': numpy.log10(norm),
            #           'intMatchedTot': numpy.log10(sum([spec[dist_mask[:,i]][:,1].sum() for i in range(dist_mask.shape[1])])),
                        'intMatchedTot': logsumI,
                        'relIntMatchedTot': sumI / norm, 'RMS': numpy.sqrt(numpy.mean([(d[m]**2).mean() for d, m in zip(dists, masks) if m.sum() != 0])),
                        'RMSppm': numpy.sqrt(numpy.mean([(((d[m] / mz_array[m]) * 1e6)**2).mean() for d, m in zip(dists, masks) if m.sum() != 0])),
                        'meanAbsFragDa': numpy.mean([d[m].mean() for d, m in zip(dists, masks) if m.sum() != 0]), 'meanAbsFragPPM': numpy.mean([(d[m] / mz_array[m]).mean() for d, m in zip(dists, masks) if m.sum() != 0]),
                        'rawscore': score,
                        'expMass': q[i]['mass'],
                        'calcMass': c['mass'],

                'score': score, 'sumI': logsumI, 'total_matched': total_matched, 'title': q[i]['title'], 'desc': c['desc'], 'seq': c['seq'], 'modseq': "".join([s if m == 0 else s + "[{}]".format(m) for s, m in zip(c['seq'], c['mods'])])})
            frag_names = list(map(lambda i: "by"[i % 2] + str(i // 2 + 1), range(len(theoretical))))
            for i in range(len(frag_names)):
                ret[-1]['tot_{}_matches'.format(frag_names[i])] = masks[i].sum() if i < len(masks) else 0
            ret[-1]['mc'] = len(re.findall(blackboard.config['processing.db']['digestion'], c['seq']))
            ret[-1]['mc0'] = ret[-1]['mc'] == 0
            ret[-1]['mc1'] = ret[-1]['mc'] == 1
            ret[-1]['mc2'] = ret[-1]['mc'] == 2

        else:
            ret.append({"dM": (c['mass'] - q[i]['mass']) / c['mass'],
                        "absdM": abs((c['mass'] - q[i]['mass']) / c['mass']),
                        "peplen": len(c['seq']),
                        "ionFrac": total_matched / len(masks),
                        #'relIntTotMatch': sumI / norm,
                        'charge': int(q[i]['charge']),
                        'z2': int(q[i]['charge'] == 2),
                        'z3': int(q[i]['charge'] == 3),
                        'z4': int(q[i]['charge'] == 4),
                        'rawscore': score,
                        #'xcorr': xcorr,
                        'expMass': q[i]['mass'],
                        'calcMass': c['mass'],

                'score': score, 'sumI': logsumI, 'total_matched': total_matched, 'title': q[i]['title'], 'desc': c['desc'], 'seq': c['seq'], 'modseq': "".join([s if m == 0 else s + "[{}]".format(m) for s, m in zip(c['seq'], c['mods'])])})
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
    m_cur = blackboard.META_CONN.cursor()

    blackboard.execute(cur, blackboard.select_str("queries", ["rowid"] + blackboard.QUERY_COLS, "WHERE rowid BETWEEN ? AND ?"), (start+1, end))
    queries = cur.fetchall()

    blackboard.execute(m_cur, "SELECT MAX(rrow) FROM meta;")
    prev_rrow = m_cur.fetchone()
    rrow = 0 if prev_rrow[0] is None else prev_rrow[0] + 1
    fname_prefix = blackboard.RES_DB_FNAME.rsplit(".", 1)[0]
    for iq, q in enumerate(queries):
        blackboard.execute(cur, blackboard.select_str("candidates", ["rowid"] + blackboard.DB_COLS, "WHERE mass BETWEEN ? AND ?"), (q['min_mass'], q['max_mass']))
        
        while True:
            cands = cur.fetchmany(batch_size)

            if len(cands) == 0:
                break

            all_q = [q] * len(cands)

            res = scoring_fn(cands, all_q)
            #r = [{'data': str({'cand_id': c['rowid'], 'query_id': q['rowid'], 'cand_mass': c['mass'], 'cand_seq': c['seq'], 'cand_mods': c['mods'], 'query_mass': q['mass'], 'query_charge': q['charge'], 'total_matched': r['total_matched'], 'sumI': r['sumI']}), 'qrow': q['rowid'], 'candrow': c['rowid'], 'score': r['score'], 'title': r['title'], 'desc': r['desc'], 'modseq': r['modseq'], 'seq': r['seq']} for r, c in zip(res, cands)]

            #r = [{'data': str(r), 'qrow': q['rowid'], 'matches': r['total_matched'], 'logSumI': r['sumI'], 'candrow': c['rowid'], 'score': r['score'], 'title': r['title'], 'desc': r['desc'], 'modseq': r['modseq'], 'seq': r['seq']} for r, c in zip(res, cands)]
            r = [{'qrow': q['rowid'], 'matches': r['total_matched'], 'logSumI': r['sumI'], 'candrow': c['rowid'], 'score': r['score'], 'title': r['title'], 'desc': r['desc'], 'modseq': r['modseq'], 'seq': r['seq'], 'query_charge': q['charge'], 'query_mass': q['mass'], 'cand_mass': c['mass'], 'rrow': rrow + ii, 'file': fname_prefix} for ii, (r, c) in enumerate(zip(res, cands)) if r['score'] > 0]
            if len(r) > 0:
                blackboard.executemany(res_cur, blackboard.maybe_insert_dict_str("results", blackboard.RES_COLS), r)
                #blackboard.RES_CONN.commit()
                metar = [{'score': r['score'], 'qrow': q['rowid'], 'candrow': c['rowid'], 'data': str(r), "rrow": rrow + ii} for ii, (r, c) in enumerate(zip(res, cands)) if r['score'] > 0]
                blackboard.executemany(m_cur, blackboard.maybe_insert_dict_str("meta", blackboard.META_COLS), metar)
                #blackboard.META_CONN.commit()
                rrow += len(res)
    blackboard.execute(res_cur, "CREATE INDEX IF NOT EXISTS res_score_qrow_idx ON results (qrow ASC, score DESC);")
    blackboard.execute(m_cur, "CREATE INDEX IF NOT EXISTS m_rrow_idx ON meta (rrow ASC);")
    blackboard.RES_CONN.commit()
    blackboard.META_CONN.commit()

def prepare_search():
    """
    Creates necessary tables in the temporary database for final search results
    """

    pass
