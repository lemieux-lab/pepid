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

    import sys
    max_exp = sys.float_info.max_exp

    for i in range(len(cands)):
        spectrum = numpy.asarray(q[i]['spec'].data)
        mz_array = spectrum[:,0]
        intens = spectrum[:,1]
        norm = intens.sum()
        charge = q[i]['charge']
        qmass = q[i]['mass']
        c = cands[i]

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
                # Must limit to float range to avoid overflow later
                mult.append(numpy.math.factorial(min(20, nmatched)))
                sumI += sumi
                score += sumi / norm

        if total_matched == 0:
            ret.append({'score': 0})
            continue

        score_l2 = numpy.ceil(numpy.log2(float(score)))
        for m in mult:
            # log2 explicitly requires the arg to be float if within a loop
            m_l2 = numpy.ceil(numpy.log2(float(m)))
            if m_l2 + score_l2 >= max_exp: # we are maxed out, bail
                score = sys.float_info.max
                break
            else:
                score *= m
                score_l2 += m_l2
        logsumI = numpy.log10(sumI)

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
                        "ionFrac": total_matched / sum(map(len, masks)),
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

    Candidates and queries are rows as saved in the DB, whose description can be found
    in `blackboard.py`.
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
            r = [{'qrow': q['rowid'], 'matches': r['total_matched'], 'logSumI': r['sumI'], 'candrow': c['rowid'], 'score': r['score'], 'title': r['title'], 'desc': r['desc'], 'modseq': r['modseq'], 'seq': r['seq'], 'query_charge': q['charge'], 'query_mass': q['mass'], 'cand_mass': c['mass'], 'rrow': rrow + ii, 'file': fname_prefix} for ii, (r, c) in enumerate(zip(res, cands)) if r['score'] > 0]
            if len(r) > 0:
                blackboard.executemany(res_cur, blackboard.maybe_insert_dict_str("results", blackboard.RES_COLS), r)
                metar = [{'score': r['score'], 'qrow': q['rowid'], 'candrow': c['rowid'], 'data': str(r), "rrow": rrow + ii} for ii, (r, c) in enumerate(zip(res, cands)) if r['score'] > 0]
                blackboard.executemany(m_cur, blackboard.maybe_insert_dict_str("meta", blackboard.META_COLS), metar)
                rrow += len(res)
    blackboard.execute(res_cur, "CREATE INDEX IF NOT EXISTS res_qrow_idx ON results (qrow ASC, score DESC);")
    blackboard.execute(res_cur, "CREATE INDEX IF NOT EXISTS res_rrow_idx ON results (rrow ASC);")
    blackboard.execute(m_cur, "CREATE INDEX IF NOT EXISTS m_rrow_idx ON meta (rrow ASC);")
    blackboard.RES_CONN.commit()
    blackboard.META_CONN.commit()

def prepare_search():
    """
    Creates necessary tables in the temporary database for final search results
    """

    pass
