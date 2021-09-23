import blackboard
import ctypes
import numpy
import fcntl
import os
import pickle

lib = ctypes.cdll.LoadLibrary("./libhelper.so")

class Query(ctypes.Structure):
    _fields_ = [("title", ctypes.c_char * 1024),
                ("rt", ctypes.c_float),
                ("charge", ctypes.c_int),
                ("mass", ctypes.c_float),
                ("npeaks", ctypes.c_int),
                ("min_mass", ctypes.c_float),
                ("max_mass", ctypes.c_float),
                ("meta", ctypes.c_char * 10240),
                ("spec", ((ctypes.c_float) * 2) * blackboard.config['search'].getint('max peaks'))]

class Db(ctypes.Structure):
    _fields_ = [("desc", ctypes.c_char * 1024),
                ("seq", ctypes.c_char * 128),
                ("mods", ctypes.c_float * 128),
                ("rt", ctypes.c_float),
                ("length", ctypes.c_uint32),
                ("npeaks", ctypes.c_uint32),
                ("mass", ctypes.c_float),
                ("meta", ctypes.c_char * 10240),
                ("spec", (ctypes.c_float * 2) * blackboard.config['search'].getint('max peaks'))]

class Res(ctypes.Structure):
    _fields_ = [('title', ctypes.c_char * 1024),
                ('description', ctypes.c_char * 1024),
                ('seq', ctypes.c_char * 128),
                ('modseq', ctypes.c_float * 128),
                ('length', ctypes.c_int),
                ('calc_mass', ctypes.c_float),
                ('mass', ctypes.c_float),
                ('rt', ctypes.c_float),
                ('charge', ctypes.c_int),
                ('score', ctypes.c_double),
                ('score_data', ctypes.c_char * 10240)]

class ScoreData(ctypes.Structure):
    _fields_ = [
                ("cands", ctypes.c_void_p),
                ("n_cands", ctypes.c_int),
                ("npeaks", ctypes.c_int),
                ("elt_size", ctypes.c_uint64),
                ("tol", ctypes.c_float),
                ("q", ctypes.c_void_p),
]

class ScoreRet(ctypes.Structure):
    _fields_ = [
                ("distances", ctypes.POINTER(ctypes.c_double)),
                ("mask", ctypes.POINTER(ctypes.c_char)),
                ("scores", ctypes.POINTER(ctypes.c_double)),
                ("sumI", ctypes.POINTER(ctypes.c_double)),
                ("total_matched", ctypes.POINTER(ctypes.c_uint32)),
                ("theoretical", ctypes.POINTER(ctypes.c_float)),
                ("spec", ctypes.POINTER(ctypes.c_float)),
                ("ncands", ctypes.c_uint32),
                ("npeaks", ctypes.c_uint32)
]

lib.rnhs.restype = ctypes.POINTER(ScoreRet)
lib.alloc.restype = ctypes.c_void_p
lib.alloc.argtypes = [ctypes.c_uint64]
lib.score_str.restype = ctypes.c_char_p
lib.score_str.argtypes = [ctypes.c_void_p]
lib.free_score.argtypes = [ctypes.c_void_p]
lib.free.argtypes = [ctypes.c_void_p]

def free(obj):
    lib.free_ptr(ctypes.cast(obj, ctypes.c_void_p))

def free_score(obj):
    lib.free_score(ctypes.pointer(obj))

def query_to_c(q):
    ret = Query()
    retptr = ctypes.pointer(ret)

    ret.title = q['title'].encode('ascii')
    ret.rt = q['rt']
    ret.charge = q['charge']
    ret.mass = q['mass']
    ret.npeaks = min(len(q['spec']), blackboard.config['search'].getint('max peaks'))
    ret.min_mass = q['min_mass']
    ret.max_mass = q['max_mass']
    #ret.meta = repr(q['meta']).encode('ascii')
    spec = q['spec']

    for i in range(min(len(spec), blackboard.config['search'].getint('max peaks'))):
        ret.spec[i][0] = spec[i][0]
        ret.spec[i][1] = spec[i][1]

    return ctypes.cast(retptr, ctypes.c_void_p)

def one_score_to_py(score, cands, q, n):
    ret = {}
    ret['distance'] = []
    ret['mask'] = []
    ret['theoretical'] = []
    ret['spec'] = []
    ret['sumI'] = score.sumI[n]
    ret['total_matched'] = score.total_matched[n]
    ret['score'] = score.scores[n]

    for i in range(len(cands[n]['spec'])):
        if cands[n]['spec'][i] != 0:
            ret['distance'].append(score.distances[n * score.npeaks + i])
            ret['mask'].append(ord(score.mask[n * score.npeaks + i]))
    ret['spec'] = q['spec']
    ret['theoretical'] = cands[n]['spec']
    return ret

def nth_score(score, cands, q, n):
    ret = {}
    ret_data = one_score_to_py(score, cands, q, n)
    ret['data'] = ret_data
    ret['score'] = ret_data['score']
    return ret

def score_to_py(scoreptr, q, cands, n_scores):
    score = scoreptr[0]
    ret = []
    for i in range(n_scores):
        s = nth_score(score, cands, q, i)
        s['title'] = q['title']
        s['desc'] = cands[i]['desc']
        s['score'] = s['score']
        s['seq'] = cands[i]['seq']
        s['modseq'] = "".join([s if m == 0 else s + "[{}]".format(m) for s,m in zip(cands[i]['seq'], cands[i]['mods'])])
        ret.append(s)
    return ret

def cands_to_c(cands):
    ret = (Db * len(cands))()
    keys = list(cands[0].keys())
    for i in range(len(cands)):
        for k in keys:
            if k not in ('spec', 'mods', 'seq', 'desc', 'meta'):
                setattr(ret[i], k, cands[i][k])
        ret[i].desc = cands[i]['desc'][:1023].encode('ascii')
        ret[i].seq = cands[i]['seq'][:127].encode('ascii')
        spec = cands[i]['spec']
        for j in range(min(len(spec), blackboard.config['search'].getint('max peaks'))):
            ret[i].spec[j][0] = spec[j]
            ret[i].spec[j][1] = 0
        for j in range(min(len(cands[i]['mods']), 127)):
            ret[i].mods[j] = cands[i]['mods'][j]
        ret[i].npeaks = min(len(cands[i]['spec']), blackboard.config['search'].getint('max peaks'))
        ret[i].length = min(len(cands[i]['seq']), 127)
    return ctypes.cast(ret, ctypes.c_void_p)

def rnhs(q, cands, tol):
    qptr = query_to_c(q)
    ccands = cands_to_c(cands)
    data = ScoreData()
    data.q = qptr
    data.tol = tol
    data.n_cands = len(cands)
    data.npeaks = blackboard.config['search'].getint('max peaks')
    data.elt_size = ctypes.sizeof(Db)
    data.cands = ccands
    ret = lib.rnhs(data)
    out = score_to_py(ret, q, cands, ret[0].ncands)
    lib.free_score(ret)
    return out

def lock():
    fcntl.lockf(blackboard.LOCK, fcntl.LOCK_EX)

def unlock():
    fcntl.lockf(blackboard.LOCK, fcntl.LOCK_UN)
