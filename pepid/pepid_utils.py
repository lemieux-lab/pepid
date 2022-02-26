import numpy

AA_TABLE = {
'G': 57.021464,
'A': 71.037114,
'S': 87.032029,
'P': 97.052764,
'V': 99.068414,
'T': 101.04768,
'C': 103.00919,
'L': 113.08406,
'I': 113.08406,
'N': 114.04293,
'D': 115.02694,
'Q': 128.05858,
'K': 128.09496,
'E': 129.04259,
'M': 131.04048,
'H': 137.05891,
'F': 147.06841,
'R': 156.10111,
'Y': 163.06333,
'W': 186.07931,
'_': 999999999.0
}

MASS_C = 12.011036905 # Average mass
MASS_N = 14.006763029 # Average mass
MASS_S = 32.064346847 # Average mass
MASS_Cl = 35.452738215 # Average mass
MASS_Br = 79.903527117 # Average mass
MASS_F = 18.998403
MASS_O = 15.994915
MASS_H = 1.0078250
MASS_OH = MASS_O + MASS_H

AMINOS = list(AA_TABLE.keys())
MASSES = [AA_TABLE[a] for a in AMINOS]

MASS_CAM = 57.0214637236 

def calc_ppm(x, y):
    #big = max(x, y)
    #small = min(x, y)
    return (numpy.abs(x - y) / float(y)) * 1e6

def calc_rev_ppm(y, ppm):
    return (ppm * 1e-6) * y

def b_series(seq, mods, nterm, cterm, z=1):
    return ((numpy.cumsum([MASSES[AMINOS.index(s)] + m for s, m in zip(seq, mods)])) + cterm - MASS_OH + (MASS_H * z)) / z

def y_series(seq, mods, nterm, cterm, z=1):
    series = (numpy.cumsum([MASSES[AMINOS.index(s)] + m for s, m in zip(seq[::-1], mods[::-1])]) + cterm + MASS_H + (MASS_H * z)) / z
    return series

def theoretical_mass(seq, mods, nterm, cterm):
    ret = sum([MASSES[AMINOS.index(s)] + m for s, m in zip(seq, mods)]) + nterm + cterm
    return ret

def neutral_mass(seq, mods, nterm, cterm, z=1):
    return (theoretical_mass(seq, mods, nterm, cterm) + (MASS_H * z)) / z

ion_shift = {
    'a': 46.00547930326002,
    'b': 18.010564683699954,
    'c': 0.984015582689949,
    'x': -25.979264555419945,
    'y': 0.0,
    'z': 17.026549101010005,
}

def theoretical_masses(seq, mods, nterm, cterm, charge=1, series="by"):
    masses = []
    cterm_generators = {"y": y_series, "x": y_series, "z": y_series}
    nterm_generators = {"b": b_series, "a": b_series, "c": b_series}
    for z in range(1, charge+1):
        for s in series:
            if s in "xyz":
                masses.append(cterm_generators[s](seq, mods, nterm=nterm, cterm=cterm, z=z).reshape((-1, 1)) - (ion_shift[s] - ion_shift['y']) / z)
            elif s in "abc":
                masses.append(nterm_generators[s](seq, mods, nterm=nterm, cterm=cterm, z=z).reshape((-1, 1)) - (ion_shift[s] - ion_shift['b']) / z)
            else:
                raise ValueError("Series '{}' not supported, available series are {}".format(s, list(cterm_generators.keys()) + list(nterm_generators.keys())))
    return masses

def import_or(s, default):
    try:
        mod, fn = s.rsplit('.', 1)
        return getattr(__import__(mod, fromlist=[fn]), fn)
    except:
        import sys
        sys.stderr.write("Could not find '{}', using default value instead\n".format(s))
        return default
