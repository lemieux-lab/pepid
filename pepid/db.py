import numpy
import glob
import re
from os import path
import blackboard
import math
import pepid_utils
import time
import os
import random
import copy
import pickle

import helper

class DbSettings():
    def __init__(self):
        pass

    def load_settings(self):
        self.digestion_pattern = blackboard.config['database']['digestion']
        self.digestion_regex = re.compile("(^|(?<={0})).*?({0}|$)".format(blackboard.config['database']['digestion']))
        self.cleaves = blackboard.config['database'].getint('max missed cleavages')
        self.min_lgt = blackboard.config['database'].getint('min length')
        self.max_lgt = blackboard.config['database'].getint('max length')
        self.min_mass = blackboard.config['database'].getfloat('min mass')
        self.max_mass = blackboard.config['database'].getfloat('max mass')
        self.var_mods = blackboard.config['search']['variable modifications'].split(",")
        self.var_mods = [[x[0], float(x[1:])] for x in self.var_mods]
        self.var_mods = {k:[x[1] for x in self.var_mods if x[0] == k] for k in set(x[0] for x in self.var_mods)}
        self.fixed_mods = blackboard.config['search']['fixed modifications'].split(",")
        self.fixed_mods = [[x[0], float(x[1:])] for x in self.fixed_mods]
        self.fixed_mods = {k:[x[1] for x in self.fixed_mods if x[0] == k] for k in set(x[0] for x in self.fixed_mods)}
        self.cterm = blackboard.config['search'].getfloat('cterm cleavage')
        self.nterm = blackboard.config['search'].getfloat('nterm cleavage')
        self.max_vars = blackboard.config['search'].getint('max variable modifications')

        self.use_decoys = blackboard.config['decoys'].getboolean('generate decoys')
        self.decoy_prefix = blackboard.config['decoys']['decoy prefix']
        self.decoy_method = blackboard.config['decoys']['decoy method']

        self.seq_types = ['normal']
        if self.use_decoys:
            self.seq_types.append('decoy')

def count_db():
    """
    Opens the protein database specified in config and
    counts how many entries it contains
    """

    f = open(blackboard.config['data']['database'])
    cnt = 0
    for l in f:
        cnt += (l[0] == '>')
    return cnt

def count_peps():
    """
    Returns how many entries exist in the processed database
    """

    return helper.count(blackboard.DB_PATH.rsplit(".bin", 1)[0] + "_index.bin", helper.KEY_TYPE)

def pred_rt(cands, n):
    """
    Dummy function for retention time prediction that just outputs 0.
    """

    ret = helper.get_buffer(helper.RT_TYPE, n)
    for i in range(n):
        ret[i] = 0.0

    return ret

def identipy_theoretical_spectrum(cands, n):
    """
    Simple spectrum prediction function generating b- and y-series ions
    without charged variants
    """

    cterm = blackboard.config['search'].getfloat('cterm cleavage')
    nterm = blackboard.config['search'].getfloat('nterm cleavage')

    ret = []

    for i in range(len(cands)):
        seq = cands[i]['sequence']
        mod = cands[i]['mods']
        th_masses = pepid_utils.identipy_theor_spectrum(seq, mod, nterm, cterm)
        ret.append(th_masses)

    return ret

def theoretical_spectrum(cands, n):
    """
    Simple spectrum prediction function generating b- and y-series ions
    without charged variants
    """

    cterm = blackboard.config['search'].getfloat('cterm cleavage')
    nterm = blackboard.config['search'].getfloat('nterm cleavage')

    ret = helper.get_buffer(helper.SPEC_TYPE, n)

    for i in range(n):
        seq = cands[i].sequence.decode('ascii')
        mod = numpy.zeros((cands[i].length,))
        for im in range(cands[i].length):
            mod[im] = cands[i].mods[im]
        th_masses = pepid_utils.theoretical_masses(seq, mod, nterm, cterm)
        for im, m in enumerate(th_masses):
            ret[i][im][0] = m
            ret[i][im][1] = 1

    return ret

def user_processing(data, n):
    """
    Parses the spectrum prediction and rt prediction functions from config
    and applies them to the candidate peptides, adding the result to the corresponding candidate
    entry in the temporary database.
    """

    rt_fn = pred_rt
    spec_fn = theoretical_spectrum
    try:
        mod, fn = blackboard.config['database']['rt function'].rsplit('.', 1)
        user_fn = getattr(__import__(mod, fromlist=[fn]), fn)
        rt_fn = user_fn
    except:
        import sys
        sys.stderr.write("[db post]: user rt prediction function not found, using default function instead\n")
    try:
        mod, fn = blackboard.config['database']['spectrum function'].rsplit('.', 1)
        user_fn = getattr(__import__(mod, fromlist=[fn]), fn)
        spec_fn = user_fn
    except:
        import sys
        sys.stderr.write("[db post]: user spectrum prediction function not found, using default function instead\n")

    arr = data
 
    rts_raw = rt_fn(arr, n)
    rts = numpy.asarray([x for x in rts_raw])
    specs_raw = spec_fn(arr, n)
    max_peaks = blackboard.config['search'].getint('max peaks')
    specs = numpy.asarray([numpy.pad(numpy.asarray([[xx[0], xx[1]] for xx in x]).reshape((-1, 2))[:max_peaks], ((0, max(0, max_peaks - len(x))), (0, 0))) for x in specs_raw])
    spec_npeaks = numpy.asarray(list(map(lambda x: numpy.where(x[:,0] == 0)[0][0], specs)))

    for i in range(n):
        arr[i].rt = rts[i]
        arr[i].npeaks = spec_npeaks[i]
        for j in range(specs.shape[0]):
            arr[i].spec[j][0] = specs[i][j][0]
            arr[i].spec[j][1] = specs[i][j][1]

    helper.free(specs_raw)
    helper.free(rts_raw)
    return arr

def process_entry(description, buff, settings):
    data = []
    if len(buff) > 0:
        for seq_type in settings.seq_types:
            if seq_type == 'decoy':
                if settings.decoy_method == 'reverse':
                    buff = buff[::-1]
                else:
                    buff = list(buff)
                    random.shuffle(buff)
                    buff = "".join(buff)
                description = settings.decoy_prefix + description

            peps = [x.group(0) for x in re.finditer(settings.digestion_regex, buff)]
            first_pep = peps[0] if len(peps) > 0 else None
            if first_pep is None:
                continue

            basic_peps_len = len(peps)
            if settings.cleaves > 0:
                for c in range(1, settings.cleaves+1):
                    for i in range(basic_peps_len-c):
                        these_peps = [peps[i]]
                        if i == 0 and peps[i][0] == 'M':
                            for x in range(1, len(peps)):
                                these_peps.append(peps[i][x:])
                        for pep in these_peps:
                            for j in range(1, c+1):
                                pep = pep + peps[i+j]
                            if settings.min_lgt <= len(pep) <= settings.max_lgt:
                                peps.append(pep)

            peps = list(filter(lambda x: settings.min_lgt <= len(x) <= settings.max_lgt, peps))
            peps = numpy.unique(peps)

            for j in range(len(peps)):
                if any([aa not in pepid_utils.AMINOS for aa in peps[j]]):
                    continue
                mods = [(sum(settings.fixed_mods[p]) if p in settings.fixed_mods else 0) for p in peps[j]]
                var_list = [(0, mods)]
                for nmods in range(settings.max_vars):
                    for iv in range(len(var_list)):
                        curr_nmods = var_list[iv][0]
                        if curr_nmods == nmods:
                            curr_mods = var_list[iv][1]
                            for mod in list(settings.var_mods.keys()):
                                for iaa, (aa, m) in enumerate(zip(peps[j], curr_mods)):
                                    if aa == mod and m == 0:
                                        var_list.append((curr_nmods+1, curr_mods[:iaa] + [sum(settings.var_mods[mod])] + curr_mods[iaa+1:]))
                      
                var_set = set(map(lambda x: tuple(x[1]), var_list)) # can't use lists in sets......
                for var in var_set:
                    mass = pepid_utils.theoretical_mass(peps[j], var, settings.nterm, settings.cterm)
                    if settings.min_mass <= mass <= settings.max_mass:
                        data.append([description, peps[j], var, 0.0, len(peps[j]), mass, repr(None)])

    return data

# XXX: TODO: in post-processing (see: sorting in search), remove duplicate peptides-mods (doing so requires peptide-order external sort, maybe use single/double argsort)
def fill_db(start, end):
    """
    Processes database entries, performing digestion and generating variable mods as needed.
    Also applies config-specified mass and length filtering.
    Data is inserted in the temporary database.

    Peptide retention time prediction and peptide spectrum prediction are generated based on config at this stage.
    """

    batch_size = blackboard.config['performance'].getint('batch size')
    input_file = open(blackboard.config['data']['database'])

    settings = DbSettings()
    settings.load_settings()

    data = []
    buff = ""
    description = ""
    entry_id = -1
    data = []
    for l in input_file:
        if l[0] == ">":
            if start <= entry_id < end:
                peps = process_entry(desc, buff, settings)
                data.extend(peps)
                       
            desc = l[1:].strip()
            entry_id += 1
            buff = ""
        else:
            buff += l.strip()
            if entry_id >= end:
                break
    if entry_id < end:
        peps = process_entry(desc, buff, settings)
        data.extend(peps)

    seqmods = [d[1] + repr(d[2]) for d in data]
    uniques = numpy.unique(seqmods, return_index=True)[-1]

    #arr = numpy.memmap(os.path.join(blackboard.config['data']['tmpdir'], "predb{}.npy".format(start)), mode="w+", dtype=blackboard.DB_DTYPE, shape=len(uniques))
    arr = numpy.zeros(dtype=blackboard.KEY_DATA_DTYPE + [('mass', 'float32')], shape=len(uniques))
    i = 0
    for k in uniques:
        d = data[k]
        if len(data[0]) == 0:
            continue
        arr[i]['seq'] = d[1]
        arr[i]['mods'] = numpy.pad(d[2][:128], (0, max(0, 128 - len(d[2]))))
        arr[i]['mass'] = d[5]
        arr[i]['desc'] = d[0]
        i += 1
    helper.dump_key(os.path.join(blackboard.config['data']['tmpdir'], "key{}.bin".format(start)), arr['mass'], offset=0, erase=True)
    helper.dump_seq(os.path.join(blackboard.config['data']['tmpdir'], "seq{}.bin".format(start)), arr[['desc', 'seq', 'mods']], offset=0, erase=True)
    del arr

def inflate(start, end):
    keys = helper.load_key(blackboard.DB_PATH.rsplit(".bin", 1)[0] + "_index.bin", start, end)
    ret, n_ret = helper.load_db(blackboard.DB_PATH, start, end)
    ret = user_processing(ret, n_ret)
    helper.dump_db(blackboard.DB_PATH, ret, n_ret, offset=start, erase=False)
    helper.free(ret)

def prepare_db():
    """
    Creates table in the temporary database which is needed for protein database processing
    """

    pass
