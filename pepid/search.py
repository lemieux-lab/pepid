import numpy
import numpy.linalg
import re
import numba
import numba.typed
import sys

if __package__ is None or __package__ == '':
    import blackboard
    import pepid_utils
else:
    from . import blackboard
    from . import pepid_utils

def typemap(x):
    if x == int:
        return "INTEGER"
    elif x == float:
        return "REAL"
    elif x == str:
        return "TEXT"
    else:
        return "BLOB"

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

def hyperscore(qcands, qs):
    """
    Sample scoring function: rnhs from identipy (with norm type = sum) or hyperscore from x!tandem (with norm type = max).
    Score output is a dictionary. If the score function will be rescored by percolator,
    the dictionary keys corresponding to the percolator parameters will be used for that purpose.
    """

    ret = []

    norm_type = blackboard.config['scoring.hyperscore']['norm type']
    ignore_weights = blackboard.config['scoring.hyperscore'].getboolean('ignore weights')

    acc_ppm = blackboard.config['scoring.hyperscore']['peak matching unit'] == 'ppm'
    acc = blackboard.config['scoring.hyperscore'].getfloat('peak matching tolerance')

    cutoff = blackboard.config['scoring.hyperscore'].getfloat('cutoff')

    match_mult = blackboard.config['scoring.hyperscore'].getfloat('match multiplier')
    disjoint = blackboard.config['scoring.hyperscore'].getboolean('disjoint model')
    max_best = blackboard.config['scoring.hyperscore'].getint('max best')
    if max_best <= 0:
        blackboard.LOG.fatal("Hyperscore: max best must be >= 0, got '{}'".format(max_best))
        sys.exit(-1)
    criterion = blackboard.config['scoring.hyperscore']['criterion best']
    type_best = blackboard.config['scoring.hyperscore']['type best']
    type_charge = type_best == 'charge'
    type_series = type_best == 'series'
    type_both = type_best == 'both'
    if (type_charge + type_series + type_both) != 1:
        blackboard.LOG.fatal("Hyperscore: type best must be exactly one of charge, series or both, got '{}'".format(type_best))
        sys.exit(-1)
    series_count = blackboard.config['scoring.hyperscore'].getint('series count')
    match_all = not blackboard.config['scoring.hyperscore'].getboolean('match only closest peak')

    for q, cands in zip(qs, qcands):
        ret.append([])

        spectrum = numpy.asarray(q['spec'].data, dtype='float32')
        mz_array = spectrum[:,0]
        intens = spectrum[:,1]
        norm = intens.sum() if norm_type == 'sum' else intens.max()
        charge = q['charge']
        qmass = q['mass']

        for i in range(len(cands)):
            c = cands[i]

            theoretical = c['spec'].data
            seqs = c['seq']
            seq_mass = c['mass']

            score = 0.0
            total_matched = 0
            sumI = 0.0

            theoretical = numpy.asarray(theoretical, dtype='float32')[:int((charge-1) * series_count)]

            sumis, series_matches = hyperscore_score(spectrum, theoretical, norm, acc, acc_ppm, cutoff, match_mult, type_charge, type_series, type_both, series_count, ignore_weights, match_all)
            sumI = float((sumis * norm).sum())

            # Select only the best N series/charges/match-lists
            selection = numpy.argsort(series_matches if criterion == 'matches' else sumis)[-max_best:]
            total_matched = int(series_matches.sum())
            series_matches = series_matches[selection]

            if series_matches.sum() == 0:
                ret[-1].append({'score': 0})
                continue

            mults = numpy.ones(selection.shape, dtype='float32')
            import sys
            float_lim = sys.float_info.max
            for j, matches in enumerate(series_matches):
                if matches > 0:
                    mults[j] = min(numpy.math.factorial(min(20, matches)), float_lim)
            if not disjoint:
                s = sumis.sum()
                score = 1
                for m in mults:
                    score *= s * m
            else:
                score = sumis.sum()
                for m in mults:
                    score *= m
            score = float(min(score, float_lim))

            logsumI = float(numpy.log10(sumI)) # note: x!tandem uses a factor 4 to multiply this by default

            ret[-1].append({"dM": (c['mass'] - q['mass']) / c['mass'],
                        "absdM": abs((c['mass'] - q['mass']) / c['mass']),
                        "peplen": len(c['seq']),
                        "ionFrac": float(total_matched / (theoretical.shape[0] * theoretical.shape[1])),
                        #'relIntTotMatch': sumI / norm,
                        'charge': int(q['charge']),
                        'z2': int(q['charge'] == 2),
                        'z3': int(q['charge'] == 3),
                        'z4': int(q['charge'] == 4),
                        'rawscore': score,
                        #'xcorr': xcorr,
                        'expMass': q['mass'],
                        'calcMass': c['mass'],
                'score': score, 'sumI': float(logsumI), 'total_matched': total_matched, 'title': q['title'], 'desc': c['desc'], 'seq': c['seq'], 'modseq': "".join([s if m == 0 else s + "[{}]".format(m) for s, m in zip(c['seq'], c['mods'])])})
    return ret

@numba.njit(locals={'spectrum': numba.float32[:,::1], 'theoretical': numba.float32[:,:,::1], 'acc': numba.float32, 'delta': numba.float32, 'series_count': numba.int32, 'norm': numba.float32, 'spec_idx': numba.int32})
def hyperscore_score(spectrum, theoretical, norm, acc=10, acc_ppm=True, cutoff=0.01, match_mult=100.0, type_charge=False, type_series=False, type_both=True, series_count=2, ignore_weights=False, match_all=False):
    mz_array = spectrum[:,0]
    intens = spectrum[:,1]

    series_matches = numpy.zeros((((len(theoretical) // series_count) if type_charge else series_count if type_series else len(theoretical)),), dtype='int32')
    sumis = numpy.zeros((series_matches.shape[0],), dtype='float32')

    delta = float(acc+1)

    idxs = numpy.zeros((len(theoretical),), dtype='int32')

    min_dists = numpy.ones((len(theoretical), spectrum.shape[0]), dtype='float32') * numpy.inf
    prev_idx = numpy.ones((len(theoretical), spectrum.shape[0]), dtype='int32') * -1
    prev_val = numpy.ones((len(theoretical), spectrum.shape[0]), dtype='float32') * 0

    for spec_idx, (mass, intensity) in enumerate(spectrum):
        next_peak = spectrum[spec_idx+1,0] if spec_idx+1 < len(spectrum) else 0
        next_lim = (next_peak - (((acc / 1e6) * next_peak) if acc_ppm else acc)) if spec_idx+1 < len(spectrum) else numpy.inf
        for s in range(len(theoretical)):
            series_idx = s % series_count
            charge_idx = s // series_count
            for i in range(idxs[s], len(theoretical[s])):
                if acc_ppm:
                    delta = ((mass - theoretical[s,i,0]) / mass) * 1e6
                else:
                    delta = (mass - theoretical[s,i,0])
                adelta = abs(delta)

                if theoretical[s,i,0] < next_lim:
                    idxs[s] = i # Record the last index that can't be within range of the next peak

                if ((not match_all) and (min_dists[s,spec_idx] <= adelta)):
                    if adelta > acc and delta < 0:
                        break
                    continue
                elif not match_all:
                    min_dists[s,spec_idx] = adelta
                if adelta <= acc:
                    normed_intens = intensity / norm
                    if normed_intens > cutoff:
                        val = normed_intens * (theoretical[s,i,1] if not ignore_weights else 1) * match_mult
                        idx = charge_idx if type_charge else series_idx if type_series else s
                        if not match_all:
                            if prev_idx[s,idx] >= 0:
                                sumis[prev_idx[s,spec_idx]] -= prev_val[s,spec_idx]
                                series_matches[prev_idx[s,spec_idx]] -= 1
                            prev_idx[s,spec_idx] = idx
                            prev_val[s,spec_idx] = val
                        sumis[idx] += val
                        series_matches[idx] += 1
                elif delta < 0: # anything else will be increasingly further away, bail
                    break

    return sumis, series_matches

def xcorr(qcands, qs):
    """
    Sample scoring function: Xcorr
    """

    bin_ppm = blackboard.config['scoring.xcorr']['bin matching unit'] == 'ppm'
    ppm_mode = blackboard.config['scoring.xcorr']['ppm mode']
    window_size = blackboard.config['scoring.xcorr'].getint('correlation window size')
    norm_windows = blackboard.config['scoring.xcorr'].getint('norm window count')
    bin_resolution_setting = blackboard.config['scoring.xcorr'].getfloat('bin resolution')
    max_mass = blackboard.config['processing.query'].getfloat('max mass')
    min_resolution = max(0, blackboard.config['scoring.xcorr'].getfloat('min bin width'))
    series_count = blackboard.config['scoring.xcorr'].getint('series count')
    ignore_weights = blackboard.config['scoring.xcorr'].getboolean('ignore weights')

    if bin_ppm and ppm_mode == 'bins':
        if min_resolution == 0:
            blackboard.LOG.fatal("Xcorr in ppm bins mode must have a non-zero min bin width, got {} (i.e. 0)".format(blackboard.config['scoring.xcorr']['min bin width']))
            sys.exit(-1)

    flank_mode = blackboard.config['scoring.xcorr']['flank mode']
    flank_length = blackboard.config['scoring.xcorr'].getint('flank length')
    if flank_length < 0:
        blackboard.LOG.fatal("Xcorr flank length must be positive, got {}".format(blackboard.config['scoring.xcorr']['flank length']))
        sys.exit(-1)
    intensity_cutoff = blackboard.config['scoring.xcorr'].getfloat('intensity cutoff')
    match_mult = blackboard.config['scoring.xcorr'].getfloat('match multiplier')

    bins = []
    if bin_ppm and ppm_mode != 'bins':
        curr_mass = max_mass
        gap = curr_mass
        while curr_mass > 0:
            bins.append(curr_mass)
            gap = max((bin_resolution_setting * 1e-6) * curr_mass, min_resolution)
            curr_mass -= gap
        bins.append(0)
        bins = bins[::-1][:-1]
    bins = numpy.asarray(bins, dtype='float32')

    ret = []

    bin_resolution = bin_resolution_setting
    for q, cands in zip(qs, qcands):
        ret.append([])

        charge = q['charge']
        spectrum = numpy.asarray(q['spec'].data, dtype='float32')
        mass = q['mass']

        if bin_ppm and ppm_mode == 'mass':
            bin_resolution = max(mass * 1e-6 * bin_resolution_setting, min_resolution)
        elif bin_ppm and ppm_mode == 'max':
            bin_resolution = max(spectrum[:,0].max() * 1e-6 * bin_resolution_setting, min_resolution)

        filtered_spectrum = spectrum[spectrum[:,0] < mass + 50]

        binned = xcorr_normalize_spec(spectrum, mass, window_size, bin_resolution, norm_windows, bin_ppm and ppm_mode == 'bins', bins, intensity_cutoff, match_mult)
        corrected = xcorr_correct_spec(binned, window_size, flank_mode, flank_length)
        if corrected is None:
            blackboard.LOG.fatal("Unrecognized flank mode '{}', aborting.".format(flank_mode))
            sys.exit(-2)

        for i in range(len(cands)):
            c = cands[i]
            frag = numpy.asarray(c['spec'].data, dtype='float32')
            # index 0 is charge 1, so charge-1 must be used
            frag = numpy.ascontiguousarray(frag[:(charge-1)*2])

            n_matches = 0
            score = 0.0
            n_series = 0
            sumI = 0.0
            total_lgt = 0

            score, sumis, series_matches = xcorr_score(frag, corrected, binned, bin_resolution, bin_ppm and ppm_mode == 'bins', bins, ignore_weights)
            sumI = float(sumis.sum())
            n_matches = series_matches.sum()

            n_series = frag.shape[0] // series_count
            total_lgt = frag.shape[1] * frag.shape[0]

            if score <= 0:
                ret[-1].append({'score': 0})
                continue
            else:
                ret[-1].append({"dM": (c['mass'] - q['mass']) / c['mass'],
                            "absdM": abs((c['mass'] - q['mass']) / c['mass']),
                            "peplen": len(c['seq']),
                            "ionFrac": n_matches / total_lgt,
                            #'relIntTotMatch': sumI / norm,
                            'charge': int(q['charge']),
                            'z2': int(q['charge'] == 2),
                            'z3': int(q['charge'] == 3),
                            'z4': int(q['charge'] == 4),
                            'rawscore': score,
                            #'xcorr': xcorr,
                            'expMass': q['mass'],
                            'calcMass': c['mass'],
                            'score': score, 'sumI': float(sumI), 'total_matched': int(n_matches), 'title': q['title'], 'desc': c['desc'], 'seq': c['seq'], 'modseq': "".join([s if m == 0 else s + "[{}]".format(m) for s, m in zip(c['seq'], c['mods'])])})
    return ret

@numba.njit(locals={'scale': numba.int32, 'i': numba.int32, 'bins': numba.float32[::1]})
def get_bin_index(mass, bins):
    if len(bins) == 0:
        return len(bins)
    if mass < bins[0]:
        return len(bins)
    if bins[0] <= mass < bins[1]:
        return 0
    scale = len(bins)
    i = 0
    while scale > 0:
        while i + scale < len(bins):
            if bins[i + scale] <= mass:
                i += scale
            else:
                break
        scale //= 2
    if (i == len(bins)-1 and bins[i] <= mass) or (bins[i] <= mass < bins[i+1]):
        return i
    else:
        return len(bins)

@numba.njit(locals={'spec': numba.float32[:,::1], 'n_windows': numba.int32, 'prec_mass': numba.float32, 'window_size': numba.int32, 'a': numba.float32, 'b': numba.float32, 'idx': numba.int32, 'window_max_intens': numba.float32, 'norm': numba.float32, 'bins': numba.float32[::1], 'match_mult': numba.float32})
def xcorr_normalize_spec(spec, prec_mass, window_size=75, bin_resolution=0.02, norm_windows=10, bin_ppm=False, bins=numpy.asarray([], dtype='float32'), intensity_cutoff=0.05, match_mult=50):
    spec_max = spec[:,0].max()
    lgt = (int(spec_max / bin_resolution) if not bin_ppm else get_bin_index(spec_max, bins)) + window_size + 1
    ret = numpy.zeros((lgt,), dtype='float32')
    flat_spec = ret.copy()

    for i in range(len(spec)):
        idx = int(spec[i,0] / bin_resolution) if not bin_ppm else get_bin_index(spec[i,0], bins)
        if idx >= len(flat_spec):
            break
        flat_spec[idx] = max(flat_spec[idx], spec[i,1])

    flat_spec = numpy.sqrt(flat_spec)

    n_windows = norm_windows
    window_size = 0
    if not bin_ppm:
        window_size = int((min(int((prec_mass + 50) / bin_resolution), int(spec[:,0].max() / bin_resolution)) / n_windows) + 1)
    else:
        window_size = int((min(get_bin_index(prec_mass + 50, bins), get_bin_index(spec[:,0].max(), bins)) / n_windows) + 1)
    max_intens = flat_spec.max()
    cutoff = intensity_cutoff * max_intens
    for i in range(n_windows):
        window_max_intens = flat_spec[window_size*i:window_size*(i+1)].max()
        if window_max_intens > 0:
            norm = match_mult / window_max_intens

            for j in range(window_size):
                idx = window_size*i+j
                if idx >= len(flat_spec):
                    break
                if flat_spec[idx] > cutoff:
                    ret[idx] = flat_spec[idx] * norm
    return ret

invsqrt2pi = 1 / numpy.sqrt(numpy.pi * 2)

@numba.njit(locals={"rs": numba.float32, 'left': numba.float32, 'right': numba.float32, 'zero': numba.float32, 'elt': numba.float32, 'window_size': numba.int32, 'invsqrt2pi': numba.float32})
def xcorr_correct_spec(spec, window_size=75, flank_mode='sides', flank_length=3):
    ret = numpy.zeros_like(spec)
    start_rs = spec[:window_size].sum()
    zero = 0
    windows = []
    rs = start_rs
    for j in range(len(spec)):
        left = (spec[j+window_size] if (j+window_size < len(spec)) else zero)
        right = (spec[j-window_size-1] if j > 2*window_size else zero)
        windows.append(rs)
        rs += left - right

    for j in range(len(spec)):
        this = (spec[j] - (windows[j] - spec[j]) / (2*window_size))
        if flank_mode == 'none':
            ret[j] += this
            continue
        if flank_mode == 'sides':
            ret[j] += this
            if j < len(spec)-1:
                ret[j+1] += this * 0.5
            if j > 0:
                ret[j-1] += this * 0.5
        elif flank_mode == 'exp':
            ret[j] += this
            for delta in range(1, min(min(j, len(spec)-j-1), flank_length+1)):
                ret[j-delta] += ((1/2)**delta) * this
                ret[j+delta] += ((1/2)**delta) * this
        elif flank_mode == 'gauss':
            ret[j] += invsqrt2pi * this
            for delta in range(1, min(min(j, len(spec)-j-1), flank_length+1)):
                ret[j-delta] += invsqrt2pi * numpy.exp(-0.5 * delta**2) * this
                ret[j+delta] += invsqrt2pi * numpy.exp(-0.5 * delta**2) * this
        else:
            return None
    return ret

@numba.njit(locals={'ret': numba.float32, 'cand': numba.float32[:,:,::1], 'corrected': numba.float32[::1], 'bin_resolution': numba.float32, 'idx': numba.int32, 'sumis': numba.float32[::1], 'nmatches': numba.int32[::1], 'bins': numba.float32[::1]})
def xcorr_score(cand, corrected, normed, bin_resolution=0.02, bin_ppm=False, bins=numpy.asarray([], dtype='float32'), ignore_weights=False):
    ret = 0
    sumis = numpy.zeros((len(cand),), dtype='float32')
    nmatches = numpy.zeros((len(cand),), dtype='int32')
    seen = set()
    for i, series in enumerate(cand):
        for peak, weight in series:
            idx = int(peak / bin_resolution) if not bin_ppm else get_bin_index(peak, bins)
            if idx >= len(corrected):
                break
            if idx in seen:
                continue
            ret += corrected[idx] * (weight if not ignore_weights else 1)
            if normed[idx] > 1e-10 or normed[idx] < -1e-10:
                sumis[i] += normed[idx]
                nmatches[i] += 1
            seen.add(idx)
    return ret, sumis, nmatches

def xcorr_hyperscore(qcands, qs):
    """
    Sample scoring function: Xcorr
    """

    ret_xcorr = xcorr(qcands, qs)
    ret_hscore = hyperscore(qcands, qs)

    ret = [[{**rh, **rx, 'xcorr': rx['score'], 'hyperscore': rh['score']} for rx, rh in zip(retx, reth)] for retx, reth in zip(ret_xcorr, ret_hscore)]
    for re in ret:
        for r in re:
            r['score'] = max(0, r['xcorr'], numpy.sqrt(numpy.log10(r['hyperscore'] + 1)), r['xcorr'] * numpy.sqrt(numpy.log10(r['hyperscore'] + 1)))

    return ret

def stub_filter(cands, q):
    return cands

def search_core(start, end):
    """
    Core search algorithm: collects and finalizes the data, then
    parses and applies the user scoring function from the config.
    The function should take two arguments: a list of candidates, and a query.

    Candidates and queries are rows as saved in the DB, whose description can be found
    in `blackboard.py`.
    """

    blackboard.init_results_db(generate=True, base_dir=blackboard.TMP_PATH)
    shard_level = blackboard.config['scoring'].getint('sharding threshold')

    scoring_fn = pepid_utils.import_or(blackboard.config['scoring']['function'], xcorr)
    select_fn = pepid_utils.import_or(blackboard.config['scoring']['candidate filtering function'], stub_filter)

    batch_size = blackboard.config['scoring'].getint('batch size')

    cur = blackboard.CONN.cursor()
    res_cur = blackboard.RES_CONN.cursor()

    blackboard.execute(cur, "SELECT rowid, * FROM queries WHERE rowid BETWEEN ? AND ?;", (start+1, end))
    queries = cur.fetchall()

    blackboard.execute(res_cur, "SELECT MAX(rrow) FROM results;")
    prev_rrow = res_cur.fetchone()[0]
    rrow = 1 if prev_rrow is None else prev_rrow + 1

    fname_prefix = blackboard.RES_DB_FNAME.rsplit(".", 1)[0]

    n_cands = 0
    cands = []
    quers = []

    columns = None

    def insert(res, cands, q, rrow):
        nonlocal cur
        nonlocal res_cur
        nonlocal shard_level
        nonlocal fname_prefix
        nonlocal columns

        this_res = []
        for ii, (r, c) in enumerate(zip(res, cands)):
            if r['score'] <= 0:
                continue
            else:
                this_res.append({'qrow': q['rowid'],  'candrow': c['rowid'], 'score': r['score'], 'title': r['title'], 'desc': r['desc'], 'modseq': r['modseq'], 'seq': r['seq'], 'query_charge': q['charge'], 'query_mass': q['mass'], 'cand_mass': c['mass'], 'rrow': rrow, 'file': fname_prefix})
                this_res[-1] = {**this_res[-1], **{("META_" + k): v for k, v in r.items()}}
                if columns is None:
                    columns = [('META_' + k, v) for k, v in r.items()]
                    blackboard.execute(res_cur, "SELECT name FROM pragma_table_info('results');")
                    header = [k[0] for k in res_cur.fetchall()]
                    for c in columns:
                        if c[0] not in header:
                            blackboard.execute(res_cur, "ALTER TABLE results ADD COLUMN {} {};".format(c[0], typemap(type(c[1]))))
                    blackboard.RES_CONN.commit()
                rrow += 1
        if len(this_res) > 0:
            blackboard.executemany(res_cur, "INSERT OR IGNORE INTO results ({}) VALUES ({});".format(",".join(this_res[-1].keys()), ",".join([":" + x for x in this_res[-1].keys()])), this_res)
            if rrow > shard_level:
                rrow = 1
                blackboard.RES_CONN.commit()
                blackboard.execute(res_cur, "CREATE INDEX IF NOT EXISTS res_row_idx ON results (rrow ASC, score DESC);")
                #blackboard.execute(res_cur, "CREATE INDEX IF NOT EXISTS res_rrow_idx ON results (rrow ASC);")
                blackboard.RES_CONN.commit()
                res_cur.close()
                blackboard.RES_CONN.close()
                blackboard.init_results_db(generate=True, base_dir=blackboard.TMP_PATH)
                res_cur = blackboard.RES_CONN.cursor()
                fname_prefix = blackboard.RES_DB_FNAME.rsplit(".", 1)[0]
                columns = None
        return rrow

    for iq, q in enumerate(queries):
        quers.append(q)
        cands.append([])
        blackboard.execute(cur, blackboard.select_str("candidates", ["rowid"] + blackboard.DB_COLS, "WHERE mass BETWEEN ? AND ?"), (q['min_mass'], q['max_mass']))
        
        while True:
            raw_set = cur.fetchmany(batch_size)
            if len(raw_set) == 0:
                if len(cands[0]) > 0:
                    res = scoring_fn(cands, quers)
                    for oq, ocands, ores in zip(quers, cands, res):
                        rrow = insert(ores, ocands, oq, rrow)
                break
            cand_set = select_fn([dict(o) for o in raw_set], q)
            if len(cand_set) == 0:
                continue
            cands[-1].extend(cand_set)
            n_cands += len(cand_set)

            if n_cands >= batch_size:
                n_cands = 0
                res = scoring_fn(cands, quers)
                for oq, ocands, ores in zip(quers, cands, res):
                    rrow = insert(ores, ocands, oq, rrow)
                cands = [[]]
                quers = [quers[-1]]

    blackboard.RES_CONN.commit()
    blackboard.execute(res_cur, "CREATE INDEX IF NOT EXISTS res_rs_idx ON results (rrow ASC, score DESC);")
    blackboard.execute(res_cur, "CREATE INDEX IF NOT EXISTS res_qs_idx ON results (qrow ASC, score DESC);")
    blackboard.RES_CONN.commit()

def prepare_search():
    """
    Creates necessary tables in the temporary database for final search results
    """

    pass
