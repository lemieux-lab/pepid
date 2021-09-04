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
lib.score_str.restype = ctypes.c_char_p
lib.score_str.argtypes = [ctypes.c_void_p]
lib.free_score.argtypes = [ctypes.c_void_p]

def free(obj):
    lib.free_ptr(ctypes.cast(obj, ctypes.c_void_p))

def free_score(obj):
    lib.free_score(ctypes.pointer(obj))

def query_to_c(q):
    retptr = lib.alloc(ctypes.sizeof(Query))

    ret = ctypes.cast(retptr, ctypes.POINTER(Query))[0]
    ret.title = q['title'].encode('ascii')
    ret.rt = q['rt']
    ret.charge = q['charge']
    ret.mass = q['mass']
    ret.npeaks = len(q['spec'])
    ret.min_mass = q['min_mass']
    ret.max_mass = q['max_mass']
    ret.meta = repr(q['meta']).encode('ascii')
    spec = q['spec']

    for i in range(min(len(spec), blackboard.config['search'].getint('max peaks'))):
        ret.spec[i][0] = spec[i][0]
        ret.spec[i][1] = spec[i][1]

    return retptr

def one_score_to_py(score, n):
    ret = {}
    ret['distance'] = []
    ret['mask'] = []
    ret['theoretical'] = []
    ret['spec'] = []
    ret['sumI'] = score.sumI[n]
    ret['total_matched'] = score.total_matched[n]
    ret['score'] = score.scores[n]

    th_done = False
    spec_done = False
    for i in range(score.npeaks):
        if score.theoretical[n * score.npeaks * 2 + i * 2] == 0:
            th_done = True
        if not th_done:
            ret['distance'].append(score.distances[n * score.npeaks + i])
            ret['mask'].append(score.mask[n * score.npeaks + i])
            ret['theoretical'].append([score.theoretical[n * score.npeaks * 2 + i * 2], score.theoretical[n * score.npeaks * 2 + i * 2 + 1]])
        if score.spec[n * score.npeaks * 2 + i * 2] == 0:
            spec_done = True
        if not spec_done:
            ret['spec'].append([score.spec[n * score.npeaks * 2 + i * 2], score.spec[n * score.npeaks * 2 + i * 2 + 1]])
        if spec_done and th_done:
            break
    return ret

def nth_score(score, n):
    ret = {}
    ret_data = one_score_to_py(score, n)
    ret['data'] = ret_data
    ret['score'] = ret_data['score']
    ret['title'] = ret_data['title']
    ret['desc'] = ret_data['desc']
    return ret

def score_to_py(scoreptr, q, cands, n_scores):
    score = scoreptr[0]
    ret = []
    for i in range(n_scores):
        s = nth_score(score, i)
        ret.append(s)
    return ret

def cands_to_c(cands):
    retptr = lib.alloc(ctypes.sizeof(Db) * len(cands))
    ret = ctypes.cast(retptr, ctypes.POINTER(Db))
    keys = list(cands[0].keys())
    for i in range(len(cands)):
        for k in keys:
            if k not in ('spec', 'mods'):
                setattr(ret[i], k, cands[i][k] if k not in ('desc', 'seq', 'meta') else (cands[i][k].encode('ascii') if k != 'meta' else pickle.dumps(cands[i]['meta'])))
        spec = cands[i]['spec']
        for j in range(min(len(spec), blackboard.config['search'].getint('max peaks'))):
            ret[i].spec[j][0] = spec[j]
            ret[i].spec[j][1] = 0
        for j in range(min(len(cands[i]['mods']), 128)):
            ret[i].mods[j] = cands[i]['mods'][j]
        ret[i].npeaks = len(cands[i]['spec'])
        ret[i].length = len(cands[i]['seq'])
    return retptr

def insert_score_sqlite(dbpath, score, score_data):
# isinstance(score, ctypes._SimpleCData)
    pass

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
    free(ccands)
    free(qptr)
    out = score_to_py(ret, q, cands, len(cands))
    lib.free_score(ret)
    return out

def lock():
    fcntl.lockf(blackboard.LOCK, fcntl.LOCK_EX)

def unlock():
    fcntl.lockf(blackboard.LOCK, fcntl.LOCK_UN)
