import blackboard
import ctypes
import numpy

KEY_TYPE = ctypes.c_float
RT_TYPE = ctypes.c_float
SPEC_TYPE = (ctypes.c_float * 2) * blackboard.config['search'].getint('max peaks')

class Seq(ctypes.Structure):
    _fields_ = [("desc", ctypes.c_char * 1024),
                ("seq", ctypes.c_char * 128),
                ("mods", ctypes.c_float * 128)]

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
    _fields_ = [("description", ctypes.c_char * 1024),
                ("sequence", ctypes.c_char * 128),
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

class Ret(ctypes.Structure):
    _fields_ = [('n', ctypes.c_ulonglong), ('start', ctypes.c_ulonglong), ('data', ctypes.c_void_p)]

class Range(ctypes.Structure):
    _fields_ = [('start', ctypes.c_longlong), ('end', ctypes.c_longlong)]

lib = ctypes.cdll.LoadLibrary("./libhelper.so")
lib.load.restype = Ret
lib.find_data.restype = Range
lib.count.restype = ctypes.c_ulonglong
lib.make_res.restype = ctypes.POINTER(Res)

def npy_data_to_c(npy_data, klass):
    ret = (klass * len(npy_data))()
    keys = list(map(lambda x: x[0], klass._fields_))
    for i in range(len(npy_data)):
        for k in keys:
            npy_res = npy_data[i][k]
            if k in ['spec']:
                arr = getattr(ret[i], k)
                for j, p in enumerate(npy_res[:npy_data[i]['npeaks']]):
                    arr[j][0] = p[0]
                    arr[j][1] = p[1]
            elif k in ['modseq', 'mods']:
                arr = getattr(ret[i], k)
                for j, p in enumerate(npy_res[:npy_data[i]['length']]):
                    arr[j] = p
            else:
                if k in ['meta', 'score_data', 'title', 'description', 'seq', 'sequence']:
                    npy_res = npy_res.encode('ascii')
                setattr(ret[i], k, npy_res)

    return ctypes.cast(ret, ctypes.c_void_p)

def c_data_to_npy(c_data, klass, dtype):
    elt_size = ctypes.sizeof(klass)
    keys = list(map(lambda x: x[0], klass._fields_))
    n_elts = c_data.n // elt_size
    ret = numpy.zeros(shape=n_elts, dtype=dtype)
    data = ctypes.cast(c_data.data, ctypes.POINTER(klass))
    for i in range(n_elts):
        for k in keys:
            c_res = getattr(data[i], k)
            if k in ['meta', 'score_data', 'title', 'description', 'seq', 'sequence']:
                c_res = c_res.decode('ascii')
            elif k in ['spec']:
                c_res = numpy.pad(numpy.array([[c_res[j][0], c_res[j][1]] for j in range(data[i].npeaks)]).reshape((-1, 2)), ((0, max(0, blackboard.config['search'].getint('max peaks') - data[i].npeaks)), (0, 0)))
            elif k in ['modseq', 'mods']:
                c_res = numpy.pad(numpy.array([c_res[j] for j in range(data[i].length)]), ((0, max(0, 128 - data[i].length))))
            ret[i][k] = c_res
    return ret

def dump_query(fname, queries, offset=0):
    q = npy_data_to_c(queries, Query)
    return lib.dump(fname.encode('ascii'), ctypes.c_ulonglong(offset * ctypes.sizeof(Query)), q, ctypes.c_ulonglong(len(queries) * ctypes.sizeof(Query)), 1) // ctypes.sizeof(Query)

def load_query(fname, start=0, end=0):
    c_data = lib.load(fname.encode('ascii'), ctypes.c_ulonglong(start * ctypes.sizeof(Query)), ctypes.c_ulonglong(ctypes.sizeof(Query)), ctypes.c_ulonglong(end - start), 0)
    ret = c_data_to_npy(c_data, Query, blackboard.QUERY_DTYPE)
    lib.free_ret(c_data)
    return ret

def load_idx(fname, start=0, end=0):
    c_data = lib.load(fname.encode('ascii'), ctypes.c_ulonglong(start * ctypes.sizeof(ctypes.c_ulonglong)), ctypes.c_ulonglong(ctypes.sizeof(ctypes.c_ulonglong)), ctypes.c_ulonglong(end - start), 0)
    ret = numpy.zeros((c_data.n // ctypes.sizeof(ctypes.c_ulonglong),), dtype='int64')
    d = ctypes.cast(c_data.data, ctypes.POINTER(ctypes.c_ulonglong))
    for i in range(c_data.n // ctypes.sizeof(ctypes.c_ulonglong)):
        ret[i] = d[i]
    lib.free_ret(c_data)
    return ret

def count(fname, klass):
    return lib.count(fname.encode('ascii'), ctypes.sizeof(klass))

def reorder(fname_idx, fname_tgt, fname_out):
    """
    Reorders fname_tgt according to the index ordering of fname_idx, outputing the result in fname_out
    """
    lib.reorder(fname_idx.encode('ascii'), fname_tgt.encode('ascii'), fname_out.encode('ascii'), ctypes.sizeof(Db), blackboard.config['performance'].getint('batch size'))

def dump_db(fname, db, n=0, offset=0, erase=False):
    return lib.dump(fname.encode('ascii'), ctypes.c_ulonglong(offset * ctypes.sizeof(Db)), db, ctypes.c_ulonglong(n * ctypes.sizeof(Db)), 1 if erase else 0) // ctypes.sizeof(Db)

def load_db(fname, start=0, end=0):
    if(end - start <= 0):
        import sys
        sys.stderr.write("WARNING: end and start are equal, reading entire file\n")
    c_data = lib.load(fname.encode('ascii'), ctypes.c_ulonglong(start * ctypes.sizeof(Db)), ctypes.c_ulonglong(ctypes.sizeof(Db)), ctypes.c_ulonglong(end - start), 0)
    return ctypes.cast(c_data.data, ctypes.POINTER(Db)), c_data.n // ctypes.sizeof(Db)

def get_buffer(klass, n):
    return (klass * n)()

def make_res(scores, score_data, q, cands):
    cscores = (ctypes.c_double * len(scores))()
    cscore_data = (ctypes.c_char_p * len(scores))()
    for i in range(len(scores)):
        cscores[i] = scores[i]
        cscore_data[i] = "".encode('ascii') #repr(score_data[i]).encode('ascii')  # WARN: VERY slow. Should be replaced by a C stub and possibly even grisu3/dragonbox
    return lib.make_res(cscores, cscore_data, q['title'].encode('ascii'), ctypes.c_float(q['mass']), ctypes.c_float(q['rt']), ctypes.c_int(q['charge']), cands, len(scores))

def dump_res(fname, res, n, offset=0):
    return lib.dump(fname.encode('ascii'), ctypes.c_ulonglong(offset * ctypes.sizeof(Res)), res, ctypes.c_ulonglong(n * ctypes.sizeof(Res)), 0) // ctypes.sizeof(Res)

def dump_key(fname, data, offset=0, erase=True):
    r = (ctypes.c_float * len(data))()
    for i in range(len(data)):
        r[i] = data[i]
    return lib.dump(fname.encode('ascii'), ctypes.c_ulonglong(offset * ctypes.sizeof(ctypes.c_float)), r, ctypes.c_ulonglong(len(data) * ctypes.sizeof(ctypes.c_float)), 1 if erase else 0) // ctypes.sizeof(ctypes.c_float)

def seq_to_c(data):
    ret = (Seq * len(data))()
    for d, r in zip(data, ret):
        r.seq = d['seq'].encode('ascii')
        for j in range(128):
            r.mods[j] = d['mods'][j]
        r.desc = d['desc'].encode('ascii')
    return ret

def dump_seq(fname, data, offset=0, erase=False):
    r = seq_to_c(data)
    return lib.dump(fname.encode('ascii'), ctypes.c_ulonglong(offset * ctypes.sizeof(Seq)), r, ctypes.c_ulonglong(data.shape[0] * ctypes.sizeof(Seq)), 1 if erase else 0) // ctypes.sizeof(Seq)

def load_key(fname, start=0, end=0):
    c_data = lib.load(fname.encode('ascii'), ctypes.c_ulonglong(start * ctypes.sizeof(ctypes.c_float)), ctypes.c_ulonglong(ctypes.sizeof(ctypes.c_float)), end - start, 0)
    c_arr = ctypes.cast(c_data.data, ctypes.POINTER(ctypes.c_float))
    ret = numpy.zeros((c_data.n // ctypes.sizeof(ctypes.c_float),), dtype='float32')
    for i in range(len(ret)):
        ret[i] = c_arr[i]
    lib.free_ret(c_data)
    return ret

def load_seq(fname, start=0, end=0):
    c_data = lib.load(fname.encode('ascii'), ctypes.c_ulonglong(start * ctypes.sizeof(Seq)), ctypes.c_ulonglong(ctypes.sizeof(Seq)), end - start, 0)
    c_arr = ctypes.cast(c_data.data, ctypes.POINTER(Seq))
    ret = numpy.zeros((c_data.n // ctypes.sizeof(Seq),), dtype=blackboard.KEY_DATA_DTYPE)
    for i in range(len(ret)):
        ret[i]['desc'] = c_arr[i].desc
        ret[i]['seq'] = c_arr[i].seq
        ret[i]['mods'] = c_arr[i].mods
    lib.free_ret(c_data)
    return ret

def load_res(fname, start=0, end=0):
    c_data = lib.load(fname.encode('ascii'), ctypes.c_ulonglong(start * ctypes.sizeof(Res)), ctypes.c_ulonglong(ctypes.sizeof(Res)), end - start, 0)
    ret = c_data_to_npy(c_data, Res, blackboard.RES_DTYPE)
    lib.free_ret(c_data)
    return ret

def sort(fnames, out, klass):
    flist = (ctypes.c_char_p * len(fnames))()
    for i in range(len(fnames)):
        flist[i] = ctypes.c_char_p(fnames[i].encode('ascii'))
    out_size = lib.merge_sort(len(fnames), flist, out.encode('ascii'), ctypes.sizeof(klass), blackboard.config['performance'].getint('sort nodes'), blackboard.config['performance'].getint('merge batch size'), blackboard.config['performance'].getint('merge nodes'))
    return out_size // ctypes.sizeof(klass)

def find(fname, klass, low, high):
    ret = lib.find_data(fname.encode('ascii'), blackboard.config['performance'].getint('search batch size'), ctypes.sizeof(klass), ctypes.c_float(low), ctypes.c_float(high))
    return ret.start, ret.end

def free(obj):
    lib.free_ptr(ctypes.cast(obj, ctypes.c_void_p))
