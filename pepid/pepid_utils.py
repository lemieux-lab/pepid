import numpy

AA_TABLE = {
'_': 999999999.0
}

MASS_H = 1.007825035
MASS_O = 15.99491463
MASS_C = 12.00000000
MASS_N = 14.0030740
MASS_S = 31.9720707
MASS_P = 30.973762
MASS_PROT = 1.00727646688
MASS_OH = MASS_O + MASS_H

AA_TABLE['G'] = MASS_C*2  + MASS_H*3  + MASS_N   + MASS_O
AA_TABLE['A'] = MASS_C*3  + MASS_H*5  + MASS_N   + MASS_O
AA_TABLE['S'] = MASS_C*3  + MASS_H*5  + MASS_N   + MASS_O*2
AA_TABLE['P'] = MASS_C*5  + MASS_H*7  + MASS_N   + MASS_O
AA_TABLE['V'] = MASS_C*5  + MASS_H*9  + MASS_N   + MASS_O
AA_TABLE['T'] = MASS_C*4  + MASS_H*7  + MASS_N   + MASS_O*2
AA_TABLE['C'] = MASS_C*3  + MASS_H*5  + MASS_N   + MASS_O   + MASS_S
AA_TABLE['L'] = MASS_C*6  + MASS_H*11 + MASS_N   + MASS_O
AA_TABLE['I'] = MASS_C*6  + MASS_H*11 + MASS_N   + MASS_O
AA_TABLE['N'] = MASS_C*4  + MASS_H*6  + MASS_N*2 + MASS_O*2
AA_TABLE['D'] = MASS_C*4  + MASS_H*5  + MASS_N   + MASS_O*3
AA_TABLE['Q'] = MASS_C*5  + MASS_H*8  + MASS_N*2 + MASS_O*2
AA_TABLE['K'] = MASS_C*6  + MASS_H*12 + MASS_N*2 + MASS_O
AA_TABLE['E'] = MASS_C*5  + MASS_H*7  + MASS_N   + MASS_O*3
AA_TABLE['M'] = MASS_C*5  + MASS_H*9  + MASS_N   + MASS_O   + MASS_S
AA_TABLE['H'] = MASS_C*6  + MASS_H*7  + MASS_N*3 + MASS_O
AA_TABLE['F'] = MASS_C*9  + MASS_H*9  + MASS_N   + MASS_O
AA_TABLE['R'] = MASS_C*6  + MASS_H*12 + MASS_N*4 + MASS_O
AA_TABLE['Y'] = MASS_C*9  + MASS_H*9  + MASS_N   + MASS_O*2
AA_TABLE['W'] = MASS_C*11 + MASS_H*10 + MASS_N*2 + MASS_O

AMINOS = list(AA_TABLE.keys())
MASSES = [AA_TABLE[a] for a in AMINOS]

MASS_CAM = 57.0214637236 

def calc_ppm(x, y):
    #big = max(x, y)
    #small = min(x, y)
    return (numpy.abs(x - y) / float(y)) * 1e6

def calc_rev_ppm(y, ppm):
    return (ppm * 1e-6) * y

def b_series(seq, mods, nterm, cterm, z=1, exclude_end=False, weights={}):
    ret = ((numpy.cumsum([MASSES[AMINOS.index(s)] + m for s, m in zip(seq, mods)])) + cterm - MASS_OH + MASS_H + (MASS_H * (z-1))) / z
    if exclude_end:
        ret = ret[:-1]
    return numpy.asarray([[mz, 1 if aa not in weights else weights[aa]] for mz, aa in zip(ret, seq)], dtype='float32')

def y_series(seq, mods, nterm, cterm, z=1, exclude_end=False, weights={}):
    ret = (numpy.cumsum([MASSES[AMINOS.index(s)] + m for s, m in zip(seq[::-1], mods[::-1])]) + cterm + MASS_H + MASS_H + (MASS_H * (z-1))) / z
    if exclude_end:
        ret = ret[:-1]
    return numpy.asarray([[mz, 1 if aa not in weights else weights[aa]] for mz, aa in zip(ret, seq[::-1])], dtype='float32')

def theoretical_mass(seq, mods, nterm, cterm):
    ret = sum([MASSES[AMINOS.index(s)] + m for s, m in zip(seq, mods)]) + nterm + cterm
    return ret

def neutral_mass(seq, mods, nterm, cterm, z=1):
    return (theoretical_mass(seq, mods, nterm, cterm) + (MASS_PROT * (z-1))) / z

ion_shift = {
    'a': 46.00547930326002,
    'b': 18.010564683699954,
    'c': 0.984015582689949,
    'x': -25.979264555419945,
    'y': 0.0,
    'z': 17.026549101010005,
}

def theoretical_masses(seq, mods, nterm, cterm, charge=1, series="by", exclude_end=False, weights={}):
    masses = []
    cterm_generators = {"y": y_series, "x": y_series, "z": y_series}
    nterm_generators = {"b": b_series, "a": b_series, "c": b_series}
    for z in range(1, charge+1):
        for s in series:
            if s in "xyz":
                masses.append(cterm_generators[s](seq, mods, nterm=nterm, cterm=cterm, z=z, exclude_end=exclude_end, weights=weights[s] if s in weights else {}))
                masses[-1][:,0] = (masses[-1][:,0] - (ion_shift[s] - ion_shift['y'])) / z
            elif s in "abc":
                masses.append(nterm_generators[s](seq, mods, nterm=nterm, cterm=cterm, z=z, exclude_end=exclude_end, weights=weights[s] if s in weights else {}))
                masses[-1][:,0] = (masses[-1][:,0] - (ion_shift[s] - ion_shift['b'])) / z
            else:
                raise ValueError("Series '{}' not supported, available series are {}".format(s, list(cterm_generators.keys()) + list(nterm_generators.keys())))

            if s in weights and "series" in weights[s]:
                masses[-1][:,1] *= weights[s]['series']
    return masses

def import_or(s, default):
    try:
        mod, fn = s.rsplit('.', 1)
        return getattr(__import__(mod, fromlist=[fn]), fn)
    except:
        import sys
        sys.stderr.write("Could not find '{}', using default value instead\n".format(s))
        return default

# Requires (matching) scores and labels (is_target) to be sorted by scores in the appropriate direction (i.e. best first, worst last)
def calc_fdr(scores, is_target):
    fdrs = []

    n_targets = 0
    n_decoys = 0

    i = 0
    while i < len(scores):
        if is_target[i]:
            n_targets += 1
        else:
            n_decoys += 1

        fdr = n_decoys / max(n_targets, 1)
        fdrs.append(min(fdr, 1))
        while is_target[i] and (i < len(scores)-1) and (scores[i] == scores[i+1]):
            n_targets += 1
            fdrs.append(min(fdr, 1))
            i += 1
        i += 1

    return fdrs

def calc_qval(scores, is_target):
    fdrs = calc_fdr(scores, is_target)
    running = fdrs[-1]
    for i in range(len(fdrs)-1, -1, -1):
        running = min(running, fdrs[i])
        fdrs[i] = running
    return fdrs

import numba

@numba.njit()
def dense_to_sparse(spec, n_max=2000):
    ret = numpy.zeros((spec.shape[0], n_max, 2), dtype='float32')
    for i in range(spec.shape[0]):
        nz = numpy.nonzero(spec[i])[0][:n_max]
        if len(nz) > 0:
            ret[i,:len(nz),:] = numpy.asarray(list(zip(nz, spec[i][nz])), dtype='float32')
    return ret

@numba.njit()
def sparse_to_dense(spec, n_max=50000):
    ret = numpy.zeros((n_max,), dtype='float32')
    for mz, intens in spec:
        ret[int(mz)] += intens
    return ret

@numba.njit(locals={'mult': numba.float32, 'size': numba.int32})
def blit_spectrum(spec, size, mult):
    spec_raw = numpy.zeros((size,), dtype='float32')
    for mz, intens in spec:
        idx = int(numpy.round(mz / mult))
        if idx >= size:
            break
        spec_raw[idx] += intens
    max = spec_raw.max()
    if max != 0:
        spec_raw /= max

    return spec_raw

def generate_pin_header(header, line):
    import blackboard

    extra_fn = blackboard.config['misc.tsv_to_pin']['user function']
    use_extra = blackboard.config['misc.tsv_to_pin'].getboolean('use extra')
    if extra_fn.strip() == '':
        extra_fn = None
    if extra_fn is not None:
        extra_fn = import_or(extra_fn, None)

    user_extra = None
    if extra_fn is not None:
        user_extra = extra_fn(header, [[line]])[0][0]

    import sqlite3
    import os
    import msgpack
    conn = sqlite3.connect("file:" + os.path.join(blackboard.TMP_PATH, line[header.index('file')] + "_meta.sqlite") + "?cache=shared", detect_types=1, uri=True) 
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT rrow, data, extra FROM meta LIMIT 1;")
    meta = cur.fetchone()

    parsed_meta = {k: v for k, v in msgpack.loads(meta['data']).items() if type(v) != str}
    if use_extra:
        extras = msgpack.loads(meta['extra'])
        parsed_meta = {**parsed_meta, **extras}
        if user_extra is not None:
            parsed_meta = {**parsed_meta, **user_extra}
        feats = sorted(list(parsed_meta.keys()))
        if 'score' in feats:
            feats.append('deltLCn')

        head = ['PSMId', 'Label', 'ScanNr']
        if 'expMass' in feats and 'calcMass' in feats:
            head.extend(['expMass', 'calcMass'])
        head.extend(feats)
        head.extend(['Peptide', 'Proteins'])

        return head

def tsv_to_pin(header, lines, start=0):
    import blackboard
    import sqlite3
    import os
    import msgpack

    score_idx = header.index('score')
    file_idx = header.index('file')
    rrow_idx = header.index('rrow')
    qrow_idx = header.index('qrow')
    seq_idx = header.index('modseq')
    title_idx = header.index('title')
    desc_idx = header.index('desc')

    prev_db = None
    prev_title = None
    cnt = 0

    max_scores = blackboard.config['misc.tsv_to_pin'].getint('max scores')
    use_extra = blackboard.config['misc.tsv_to_pin'].getboolean('use extra')
    decoy_prefix = blackboard.config['processing.db']['decoy prefix']

    prev_db = None

    conn = None
    cur = None

    feats = None
    scores = []

    out_lines = []

    newlines = []
    for qlines in lines:
        if len(qlines) == 0:
            break
        idxs = numpy.argsort([float(line[score_idx]) for line in qlines])[::-1][:max_scores]
        newlines.append([qlines[idx] for idx in idxs])
    lines = newlines

    extra_fn = blackboard.config['misc.tsv_to_pin']['user function']
    if extra_fn is not None:
        extra_fn = import_or(extra_fn, None)

    user_extra = None
    if extra_fn is not None:
        user_extra = extra_fn(header, lines)

    for il, qlines in enumerate(lines):
        if len(qlines) == 0:
            break
        out_lines.append([])
        if conn is not None:
            cur.close()
            conn.close()
        conn = sqlite3.connect("file:" + os.path.join(blackboard.TMP_PATH, qlines[0][file_idx] + "_meta.sqlite") + "?cache=shared", detect_types=1, uri=True) 
        cur = conn.cursor()

        cur.execute("SELECT rrow, data, extra FROM meta WHERE rrow in ({}) ORDER BY score DESC;".format(",".join([line[rrow_idx] for line in qlines])))
        metas = cur.fetchall()
        parsed_metas = []

        for i, (rrow, m, e) in enumerate(metas):
            parsed_metas.append({k: v for k, v in msgpack.loads(m).items() if type(v) != str})
            if use_extra:
                extras = msgpack.loads(e)
                parsed_metas[-1] = {**parsed_metas[-1], **extras}
            if user_extra is not None:
                parsed_metas[-1] = {**parsed_metas[-1], **user_extra[il][i]}
            if feats is None:
                feats = sorted(list(parsed_metas[-1].keys()))
                if 'score' in feats:
                    feats.append('deltLCn')
            if 'score' in feats: # XXX: HACK for comet-like deltLCn feature should depend on config...
                scores.append(parsed_metas[-1]['score'])

        for j, m in enumerate(parsed_metas):
            if 'score' in feats:
                m['deltLCn'] = (m['score'] - numpy.min(scores)) / (m['score'] if m['score'] != 0 else 1)

            extraVals = [0.0, 0.0]
            if 'expMass' in m and 'calcMass' in m:
                extraVals = [m['expMass'], m['calcMass']]

            out_lines[-1].append(list(map(str, [qlines[j][title_idx], (1 - qlines[j][desc_idx].startswith(decoy_prefix)) * 2 - 1, start+il, *extraVals])))

            out_lines[-1][-1].extend(list(map(lambda k: numpy.format_float_positional(m[k], trim='0', precision=12), feats))) # percolator breaks if too many digits are provided

            out_lines[-1][-1].extend(["-." + qlines[j][seq_idx] + ".-", qlines[j][desc_idx]])
    return out_lines
