import numpy
import glob
import re
from os import path
import math
import time
import os
import random
import copy
import msgpack
import pickle

if __package__ is None or __package__ == '':
    from pepid import blackboard
    from pepid import pepid_utils
else:
    from . import blackboard
    from . import pepid_utils

class DbSettings():
    def __init__(self):
        pass

    def load_settings(self):
        self.digestion_pattern = blackboard.config['processing.db']['digestion']
        self.digestion_regex = re.compile("(^|(?<={0})).*?({0}|$)".format(blackboard.config['processing.db']['digestion']))
        self.cleaves = blackboard.config['processing.db'].getint('max missed cleavages')
        self.min_lgt = blackboard.config['processing.db'].getint('min length')
        self.max_lgt = blackboard.config['processing.db'].getint('max length')
        self.min_mass = blackboard.config['processing.db'].getfloat('min mass')
        self.max_mass = blackboard.config['processing.db'].getfloat('max mass')
        self.var_mods = blackboard.config['processing.db']['variable modifications'].split(",")
        self.var_mods = [[x[0], float(x[1:])] for x in self.var_mods]
        self.var_mods = {k:[x[1] for x in self.var_mods if x[0] == k] for k in set(x[0] for x in self.var_mods)}
        self.fixed_mods = blackboard.config['processing.db']['fixed modifications'].split(",")
        self.fixed_mods = [[x[0], float(x[1:])] for x in self.fixed_mods]
        self.fixed_mods = {k:[x[1] for x in self.fixed_mods if x[0] == k] for k in set(x[0] for x in self.fixed_mods)}
        self.cterm = blackboard.config['processing.db'].getfloat('cterm cleavage')
        self.nterm = blackboard.config['processing.db'].getfloat('nterm cleavage')
        self.max_vars = blackboard.config['processing.db'].getint('max variable modifications')

        self.use_decoys = blackboard.config['processing.db'].getboolean('generate decoys')
        self.decoy_prefix = blackboard.config['processing.db']['decoy prefix']
        self.decoy_method = blackboard.config['processing.db']['decoy method']
        self.decoy_type = blackboard.config['processing.db']['decoy type']

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

MODEL = None
class post_ml_spectrum(object):
    required_fields = {'candidates': ['seq', 'mods']}

    def __new__(cls, cands):
        if __package__ is None or __package__ == '':
            import pepid
            from pepid.ml import specgen as specgen
        else:
            from .ml import specgen as specgen
        import torch

        batch_size = blackboard.config['processing.db.post_ml_spectrum'].getint('batch size')
        device = blackboard.config['processing.db.post_ml_spectrum']['device']
        if device is None:
            device = 'cpu'

        if 'cuda' in device:
            blackboard.lock()
            gpu_lock = blackboard.acquire_lock(device)

            blackboard.lock(gpu_lock)

            blackboard.unlock()

        global MODEL

        if MODEL is None:
            MODEL = specgen.Model().eval().to(device)
            MODEL.load_state_dict(torch.load(blackboard.here("ml/best_specgen.pkl"), map_location=device))

        cterm = blackboard.config['processing.db'].getfloat('cterm cleavage')
        nterm = blackboard.config['processing.db'].getfloat('nterm cleavage')

        payloads = []
        for c in cands:
            payloads.append({'pep': c['seq'], 'mods': msgpack.loads(c['mods']), 'mass': pepid_utils.neutral_mass(c['seq'], c['mods'], nterm, cterm, z=1)})
        embs = torch.FloatTensor(specgen.embed_all(payloads))

        max_peaks = 500
        out = [None for _ in range(len(embs))]

        with torch.no_grad():
            for bidx in range(0, len(embs), batch_size):
                pred = MODEL(embs[bidx:bidx+batch_size].to(device)).detach().cpu().numpy()
                sparsed = pepid_utils.dense_to_sparse(pred.reshape((-1, pred.shape[-1])), n_max = max_peaks)
                sparsed = sparsed.reshape((-1, pred.shape[1], sparsed.shape[-2], 2))
                out[bidx:bidx+batch_size] = sparsed

        if 'cuda' in device:
            import gc
            del MODEL
            MODEL = None
            gc.collect()
            torch.cuda.empty_cache()
            blackboard.unlock(gpu_lock)

        for o in range(len(out)):
            cands[o]['meta'] = pickle.dumps({'MLSpec': out[o]})
        return cands

class theoretical_spectrum(object):
    required_fields = {'candidates': ['seq', 'mods']}

    def __new__(cls, cands):
        """
        Spectrum prediction function generating b- and y-series ions
        with charged variants
        """

        cterm = blackboard.config['processing.db'].getfloat('cterm cleavage')
        nterm = blackboard.config['processing.db'].getfloat('nterm cleavage')
        max_charge = blackboard.config['processing.db'].getint('max charge')
        exclude_end = blackboard.config['processing.db.theoretical_spectrum'].getboolean('exclude last aa')
        weights = eval(blackboard.config['processing.db.theoretical_spectrum']['weights'])

        ret = []

        for cand in cands:
            seq = cand['seq']
            mod = cand['mods']
            masses = pepid_utils.theoretical_masses(seq, mod, nterm, cterm, charge=max_charge, exclude_end=exclude_end, weights=weights)
            ret.append(masses)

        return ret

def stub(x):
    return x

def user_processing(start, end):
    """
    Parses the spectrum prediction and rt prediction functions from config
    and applies them to the candidate peptides, adding the result to the corresponding candidate
    entry in the temporary database.
    """

    import msgpack

    cur = blackboard.CONN.cursor()
    max_charge = blackboard.config['processing.db'].getint('max charge')

    rt_fn = pepid_utils.import_or(blackboard.config['processing.db']['rt function'].strip(), pred_rt)
    spec_fn = pepid_utils.import_or(blackboard.config['processing.db']['spectrum function'].strip(), theoretical_spectrum)
    user_fn = pepid_utils.import_or(blackboard.config['processing.db']['postprocessing function'].strip(), stub)

    rows = set()
    rows.add('rowid')
    rows.update(getattr(user_fn, 'required_fields', {}).get('candidates', []))
    rows.update(getattr(rt_fn, 'required_fields', {}).get('candidates', []))
    rows.update(getattr(spec_fn, 'required_fields', {}).get('candidates', []))

    blackboard.execute(cur, "SELECT {} FROM candidates WHERE rowid BETWEEN ? AND ?;".format(",".join(rows)), (start+1, end))
    data = cur.fetchall()
    data = list(map(dict, data))

    rts = rt_fn(data)
    specs = spec_fn(data)

    specs = [pickle.dumps(spec) for spec in specs]
    
    for i in range(len(data)):
        data[i]['spec'] = specs[i]
        data[i]['rt'] = rts[i]
    ret = user_fn(data)

    blackboard.executemany(cur, "UPDATE candidates SET {} WHERE rowid = :rowid;".format(",".join(["{} = :{}".format(k, k) for k in ret[0].keys()])), ret)
    #blackboard.executemany(cur, "REPLACE INTO candidates ({}) VALUES ({});".format(",".join(blackboard.DB_COLS), ",".join(list(map(lambda x: ":" + x, blackboard.DB_COLS)))), ret)
    blackboard.commit()

def process_entry(description, buff, settings):
    return process_entry_core(description, buff, settings, 'normal')

def process_entry_decoy(description, buff, settings):
    return process_entry_core(description, buff, settings, 'decoy')

def process_entry_core(description, buff, settings, seq_type):
    data = []
    if len(buff) > 0:
        is_pep_reverse = is_pep_shuffle = is_pep_decoy = False

        if seq_type == 'decoy':
            if settings.decoy_type == 'protein':
                if settings.decoy_method == 'reverse':
                    buff = buff[::-1]
                else:
                    buff = list(buff)
                    random.shuffle(buff)
                    buff = "".join(buff)
            else:
                is_pep_decoy = True
                if settings.decoy_method == 'reverse':
                    is_pep_reverse = True
                else:
                    is_pep_shuffle = True

            description = settings.decoy_prefix + description

        peps = [x.group(0) for x in re.finditer(settings.digestion_regex, buff) if len(x.group(0)) > 1]
        if is_pep_decoy:
            # NOTE: we want to preserve last AA
            if is_pep_reverse:
                peps = [pep[::-1][1:] + pep[-1] for pep in peps]
            else:
                new_peps = []
                for pep in peps:
                    new = list(pep[:-1])
                    random.shuffle(nnew)
                    new_peps.append("".join(new) + pep[-1])
                peps = new_peps

        basic_peps_len = len(peps)
        if basic_peps_len == 0:
            return []

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
                    data.append({"desc": description.split(" ", 1)[0], "decoy": seq_type == "decoy", "seq": peps[j], "mods": msgpack.dumps(var), "rt": 0.0, "length": len(peps[j]), "mass": mass, "spec": pickle.dumps(None), 'meta': msgpack.dumps(None)})
    return data

def fill_db(start, end, seq_type):
    """
    Processes database entries, performing digestion and generating variable mods as needed.
    Also applies config-specified mass and length filtering.
    Data is inserted in the temporary database.

    Peptide retention time prediction and peptide spectrum prediction are generated based on config at this stage.
    """

    batch_size = blackboard.config['processing.db'].getint('batch size')
    input_file = open(blackboard.config['data']['database'])
    if seq_type == 'decoy':
        process_protein_fn = pepid_utils.import_or(blackboard.config['processing.db']['decoy protein processing function'].strip(), process_entry_decoy)
    else:
        process_protein_fn = pepid_utils.import_or(blackboard.config['processing.db']['protein processing function'].strip(), process_entry)

    settings = DbSettings()
    settings.load_settings()

    data = []
    buff = ""
    description = ""
    entry_id = -1
    data = []

    input_file.seek(0)
    for l in input_file:
        if l[0] == ">":
            if start <= entry_id < end:
                peps = process_protein_fn(desc, buff, settings)
                data.extend(peps)
                       
            desc = l[1:].strip()
            entry_id += 1
            buff = ""
        else:
            buff += l.strip()
            if entry_id >= end:
                break
    if entry_id < end:
        peps = process_protein_fn(desc, buff, settings)
        data.extend(peps)

    cur = blackboard.CONN.cursor()

    blackboard.executemany(cur, blackboard.maybe_insert_dict_extra_str("candidates", blackboard.DB_COLS, "ON CONFLICT(seq, mods, decoy) DO UPDATE SET desc=desc || ';' || excluded.desc"), data)
    cur.close()
    blackboard.commit()

def prepare_db():
    """
    Creates table in the temporary database which is needed for protein database processing
    """

    cur = blackboard.CONN.cursor()
    blackboard.execute(cur, "DROP INDEX IF EXISTS cand_mass_idx;")
    blackboard.execute(cur, "DROP TABLE IF EXISTS candidates;")
    blackboard.execute(cur, blackboard.create_table_str("c.candidates", blackboard.DB_COLS, blackboard.DB_TYPES, extra=["UNIQUE(seq, mods, decoy)", "UNIQUE(seq, mods)"]))
    cur.close()
    blackboard.commit()
