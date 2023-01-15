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
