import blackboard
import ctypes
import numpy
import fcntl
import os
import pickle
import array

lib = ctypes.cdll.LoadLibrary("./libhelper.so")

class Query(ctypes.Structure):
    _fields_ = [
                ("npeaks", ctypes.c_int),
                ("spec", (ctypes.POINTER(ctypes.c_float)))]

class Db(ctypes.Structure):
    _fields_ = [
                ("npeaks", ctypes.POINTER(ctypes.c_uint32)),
                ("spec", ctypes.POINTER(ctypes.c_float)),
                ("valid_series", ctypes.POINTER(ctypes.c_char))]

class ScoreData(ctypes.Structure):
    _fields_ = [
                ("cands", ctypes.c_void_p),
                ("n_cands", ctypes.c_int),
                ("npeaks", ctypes.c_int),
                ("n_series", ctypes.c_int),
                ("tol", ctypes.c_float),
                ("ppm", ctypes.c_ubyte),
                ("q", ctypes.c_void_p),
]

class ScoreRet(ctypes.Structure):
    _fields_ = [
                ("distances", ctypes.POINTER(ctypes.c_double)),
                ("mask", ctypes.POINTER(ctypes.c_ubyte)),
                ("score", (ctypes.c_double)),
                ("sumI", (ctypes.c_double)),
                ("total_matched", (ctypes.c_uint32)),
]

lib.rnhs.restype = ctypes.POINTER(ScoreRet)
lib.alloc.restype = ctypes.c_void_p
lib.alloc.argtypes = [ctypes.c_uint64]
#lib.score_str.restype = ctypes.c_char_p
#lib.score_str.argtypes = [ctypes.c_void_p]
lib.free_score.argtypes = [ctypes.c_void_p]
lib.free.argtypes = [ctypes.c_void_p]

def free(obj):
    lib.free_ptr(ctypes.cast(obj, ctypes.c_void_p))

def free_score(obj):
    lib.free_score(ctypes.pointer(obj))

def queries_to_c(q):
    ret = (Query * len(q))()
    retptr = ctypes.pointer(ret)

    for i in range(len(q)):
        spec = q[i]['spec'].data[:blackboard.config['search'].getint('max peaks')]

        ret[i].npeaks = len(spec)
        ret[i].spec = ctypes.cast((ctypes.c_float * len(spec)).from_buffer(array.array('f', [x for y in spec for x in y])), ctypes.POINTER(ctypes.c_float))

    return ctypes.cast(retptr, ctypes.c_void_p)

def one_score_to_py(score, cands, q, n):
    max_charge = blackboard.config['search'].getint('max charge')
    ret = {}
    ret['distance'] = []
    ret['mask'] = []
    ret['theoretical'] = []
    ret['spec'] = []
    ret['sumI'] = score[n].sumI
    ret['total_matched'] = score[n].total_matched
    ret['score'] = score[n].score

    npeaks = blackboard.config['search'].getint('max peaks')

    distances = numpy.ctypeslib.as_array(score[n].distances, shape=(max_charge*2, npeaks))
    masks = numpy.ctypeslib.as_array(score[n].mask, shape=(max_charge*2, npeaks)).astype('int32')
    
    ret['distance'] = distances[:q[n]['charge']*2,:len(q[n]['spec'].data)].tolist()
    ret['mask'] = masks[:q[n]['charge']*2,:len(q[n]['spec'].data)].tolist()
    ret['spec'] = q[n]['spec']
    ret['theoretical'] = cands[n]['spec']
    return ret

def nth_score(score, cands, q, n):
    ret = {}
    ret_data = one_score_to_py(score, cands, q, n)
    ret['data'] = ret_data
    ret['score'] = ret_data['score']
    #ret['score'] = 0
    return ret

def score_to_py(scoreptr, q, cands, n_scores):
    score = scoreptr
    ret = []
    for i in range(n_scores):
        s = nth_score(score, cands, q, i)
        s['title'] = q[i]['title']
        s['desc'] = cands[i]['desc']
        s['seq'] = cands[i]['seq']
        s['modseq'] = "".join([s if m == 0 else s + "[{}]".format(m) for s,m in zip(cands[i]['seq'], cands[i]['mods'])])
        ret.append(s)
    return ret

def cands_to_c(cands, q_charges):
    max_charge = blackboard.config['search'].getint('max charge')
    max_peaks = blackboard.config['search'].getint('max peaks')
    ret = (Db * len(cands))()
    for i in range(len(cands)):
        spec = cands[i]['spec'].data
        ret[i].npeaks = ctypes.cast((ctypes.c_uint32 * len(spec)).from_buffer(array.array('I', [len(sp) for sp in spec])), ctypes.POINTER(ctypes.c_uint32))
        ret[i].valid_series = ctypes.cast((ctypes.c_char * (max_charge * 2)).from_buffer(array.array('b', [0 if ((k // 2 + 1) >= q_charges[i]) else 1 for k in range(len(spec))])), ctypes.POINTER(ctypes.c_char))
        specp = [0.0] * (len(spec) * max_peaks)
        for s in range(len(spec)):
            specp[s * max_peaks:s * max_peaks + len(spec[s])] = spec[s]
        ret[i].spec = ctypes.cast((ctypes.c_float * len(specp)).from_buffer(array.array('f', specp)), ctypes.POINTER(ctypes.c_float))

    return ctypes.cast(ret, ctypes.c_void_p), specp

def rnhs(q, cands, tol, ppm, whole_data=True):
    qptr = queries_to_c(q)
    ccands, _ = cands_to_c(cands, [qq['charge'] for qq in q]) # charge goes from 1 to prec_charge-1 inclusive
    data = ScoreData()
    data.q = qptr
    data.tol = tol
    data.ppm = 1 if ppm else 0
    data.n_cands = len(cands)
    data.npeaks = blackboard.config['search'].getint('max peaks')
    data.n_series = blackboard.config['search'].getint('max charge') * 2
    data.cands = ccands
    ret = lib.rnhs(data)
    if whole_data:
        out = score_to_py(ret, q, cands, len(cands))
    else:
        out = [ret[i].score for i in range(len(cands))]
    lib.free_score(ret)
    return out

def lock():
    fcntl.lockf(blackboard.LOCK, fcntl.LOCK_EX)

def unlock():
    fcntl.lockf(blackboard.LOCK, fcntl.LOCK_UN)
