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
import helper
import pickle

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

    cur = blackboard.CONN.cursor()
    blackboard.execute(cur, "SELECT COUNT(*) FROM candidates;")
    return cur.fetchone()[0]

def pred_rt(cands):
    """
    Dummy function for retention time prediction that just outputs 0.
    """

    return [0.0] * len(cands)

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

def theoretical_spectrum(cands):
    """
    Simple spectrum prediction function generating b- and y-series ions
    without charged variants
    """

    cterm = blackboard.config['search'].getfloat('cterm cleavage')
    nterm = blackboard.config['search'].getfloat('nterm cleavage')

    ret = []

    for i in range(len(cands)):
        seq = cands[i]['seq']
        mod = cands[i]['mods']
        th_masses = pepid_utils.theoretical_masses(seq, mod, nterm, cterm)
        ret.append(th_masses.tolist())

    return ret

def user_processing(start, end):
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

    cur = blackboard.CONN.cursor()
    blackboard.execute(cur, blackboard.select_str("candidates", blackboard.DB_COLS, "WHERE rowid BETWEEN ? AND ?"), (start+1, end))

    ret = cur.fetchall()
    data = [{k:(v if k not in ('spec', 'mods', 'meta') else pickle.loads(v)) for k, v in zip(blackboard.DB_COLS, results)} for results in ret]

    rts = rt_fn(data)
    specs = spec_fn(data)
    max_peaks = blackboard.config['search'].getint('max peaks')

    specs = [pickle.dumps(spec[:min(len(spec), max_peaks)]) for spec in specs]
    rowids = list(range(start+1, end+1))

    blackboard.executemany(cur, "UPDATE candidates SET rt = ?, spec = ? WHERE rowid = ?;", list(zip(rts, specs, rowids)))
    blackboard.commit()

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
                        data.append({"desc": description, "seq": peps[j], "mods": var, "rt": 0.0, "length": len(peps[j]), "mass": mass, "spec": None})
    return data

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

    cur = blackboard.CONN.cursor()
    #ipt_data = [tuple([((row[k] if k not in ('spec', 'mods', 'meta') else pickle.dumps(row[k])) if k != 'meta' else pickle.dumps(None)) for k in blackboard.DB_COLS]) for row in data]
    ipt_data = []
    for row in data:
        ipt_data.append([])
        for k in blackboard.DB_COLS:
            if k == 'meta':
                ipt_data[-1].append(pickle.dumps(None))
            elif k in ('spec', 'mods'):
                ipt_data[-1].append(pickle.dumps(row[k]))
            else:
                ipt_data[-1].append(row[k])
        ipt_data[-1] = tuple(ipt_data[-1])
        
    blackboard.executemany(cur, blackboard.insert_all_str("candidates", blackboard.DB_COLS), ipt_data)
    cur.close()
    blackboard.commit()

def prepare_db():
    """
    Creates table in the temporary database which is needed for protein database processing
    """

    cur = blackboard.CONN.cursor()
    blackboard.execute(cur, "DROP INDEX IF EXISTS cand_mass_idx;")
    blackboard.execute(cur, "DROP TABLE IF EXISTS candidates;")
    blackboard.execute(cur, blackboard.create_table_str("c.candidates", blackboard.DB_COLS, blackboard.DB_TYPES, ["UNIQUE(seq, mods) ON CONFLICT IGNORE"]))
    cur.close()
    blackboard.commit()
