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

def b_series(seq, mods, cterm, nterm, z=1):
    return ((numpy.cumsum([MASSES[AMINOS.index(s)] + m for s, m in zip(seq, mods)])) + (MASS_H * z)) / z

def y_series(seq, mods, cterm, nterm, z=1):
    series = (numpy.cumsum([MASSES[AMINOS.index(s)] + m for s, m in zip(seq[::-1], mods[::-1])]) + nterm + cterm + (MASS_H * z)) / z
    return series

def filter_by_mass(candidates, mass, tol, tol_ppm):
    if tol_ppm:
        dist = calc_rev_ppm(tol, mass)
        left = numpy.searchsorted(candidates['mass'], mass - dist, side='left')
        right = numpy.searchsorted(candidates['mass'], mass + dist, side='right') + 1
    else:
        left = numpy.searchsorted(candidates['mass'], mass - tol, side='left')
        right = numpy.searchsorted(candidates['mass'], mass + tol, side='right') + 1
    return left, right

def theoretical_mass(seq, mods, nterm, cterm):
    ret = sum([MASSES[AMINOS.index(s)] + m for s, m in zip(seq, mods)]) + nterm + cterm
    return ret

def identipy_theoretical_masses(seq, mods, nterm, cterm, charge=1, series="by"):
    import pyteomics
    from pyteomics import electrochem

    peptide = seq
    nm = theoretical_mass(seq, mods, nterm, cterm)
    peaks = []
    pl = len(peptide) - 1
    for c in range(1, charge + 1):
        for ion_type in series:
            nterminal = ion_type in 'abc'
            maxmass = identipy_calc_ions_from_neutral_mass(peptide, nm, ion_type=ion_type, charge=c, cterm_mass=cterm, nterm_mass=nterm)
            if nterminal:
                marr = identipy_get_n_ions(peptide, mods, maxmass, pl, c)
            else:
                marr = identipy_get_c_ions(peptide, mods, maxmass, pl, c)

            marr = numpy.asarray(marr)
            peaks.append(marr.reshape((-1, 1)))
    return peaks

def theoretical_masses(seq, mods, nterm, cterm, charge=1, series="by"):
    masses = []
    cterm_generators = {"y": y_series}
    nterm_generators = {"b": b_series}
    for z in range(1, charge+1):
        for s in series:
            if s in "xyz":
                masses.append(cterm_generators[s](seq, mods, nterm=nterm, cterm=cterm, z=z))
            elif s in "abc":
                masses.append(nterm_generators[s](seq, mods, nterm=nterm, cterm=cterm, z=z))
            else:
                raise ValueError("Series '{}' not supported, available series are {}".format(s, list(cterm_generators.keys()) + list(nterm_generators.keys())))
    masses = numpy.vstack(masses)
    return masses

def identipy_get_n_ions(peptide, mods, maxmass, pl, charge):
    tmp = [maxmass, ]
    for i in range(1, pl):
        tmp.append(tmp[-1] - (MASSES[AMINOS.index(peptide[-i-1])] + mods[-i-1])/charge)
    return tmp

def identipy_get_c_ions(peptide, mods, maxmass, pl, charge):
    tmp = [maxmass, ]
    for i in range(pl-2, -1, -1):
        tmp.append(tmp[-1] - (MASSES[AMINOS.index(peptide[-(i+2)])] + mods[-(i+2)])/charge)
    return tmp

def identipy_theor_spectrum(seq, mods, nterm_mass, cterm_mass, types=['b', 'y'], maxcharge=None):
    import pyteomics
    from pyteomics import electrochem

    peptide = seq
    if not maxcharge:
        maxcharge = 1 + int(pyteomics.electrochem.charge(peptide, pH=2))
    nm = theoretical_mass(seq, mods, nterm_mass, cterm_mass)
    peaks = {}
    pl = len(peptide) - 1
    for charge in range(1, maxcharge + 1):
        for ion_type in types:
            nterminal = ion_type in 'abc'
            maxmass = identipy_calc_ions_from_neutral_mass(peptide, nm, ion_type=ion_type, charge=charge, cterm_mass=cterm_mass, nterm_mass=nterm_mass)
            if nterminal:
                marr = identipy_get_n_ions(peptide, mods, maxmass, pl, charge)
            else:
                marr = identipy_get_c_ions(peptide, mods, maxmass, pl, charge)

            marr = numpy.asarray(marr)
            peaks[(ion_type, charge)] = marr.reshape((marr.shape[0], 1)).tolist()
    return peaks

identipy_ion_shift_dict = {
    'a': 46.00547930326002,
    'b': 18.010564683699954,
    'c': 0.984015582689949,
    'x': -25.979264555419945,
    'y': 0.0,
    'z': 17.026549101010005,
}

def identipy_calc_ions_from_neutral_mass(peptide, nm, ion_type, charge, cterm_mass, nterm_mass):
    if ion_type in 'abc':
        nmi = nm - MASSES[AMINOS.index(peptide[-1])] - identipy_ion_shift_dict[ion_type] - (cterm_mass - 17.002735)
    else:
        nmi = nm - MASSES[AMINOS.index(peptide[0])] - identipy_ion_shift_dict[ion_type] - (nterm_mass - 1.007825)
    return (nmi + 1.0072764667700085 * charge) / charge 
