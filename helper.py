import blackboard
import ctypes
import numpy

class Seq(ctypes.Structure):
    _fields_ = [("seq", ctypes.c_char * 128),
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
    _fields_ = [('n', ctypes.c_ulonglong), ('data', ctypes.c_void_p)]

class Range(ctypes.Structure):
    _fields_ = [('start', ctypes.c_longlong), ('end', ctypes.c_longlong)]

lib = ctypes.cdll.LoadLibrary("./libhelper.so")
lib.load.restype = Ret
lib.find_data.restype = Range
lib.count.restype = ctypes.c_ulonglong

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
    return lib.dump(fname.encode('ascii'), offset * ctypes.sizeof(Query), q, len(queries) * ctypes.sizeof(Query), 1) // ctypes.sizeof(Query)

def load_query(fname, start=0, end=0):
    c_data = lib.load(fname.encode('ascii'), start * ctypes.sizeof(Query), (end - start) * ctypes.sizeof(Query), 0)
    ret = c_data_to_npy(c_data, Query, blackboard.QUERY_DTYPE)
    return ret

def count(fname, klass):
    return lib.count(fname.encode('ascii'), ctypes.sizeof(klass))

def dump_db(fname, db, offset=0, erase=False):
    d = npy_data_to_c(db, Db)
    dret = Ret()
    dret.n = len(db) * ctypes.sizeof(Db)
    dret.data = d
    rd = c_data_to_npy(dret, Db, blackboard.DB_DTYPE)
    import sys
    try:
        c = (rd == db).all()
        if not c:
            sys.stderr.write("Conversion failure!\n")
    except:
        sys.stderr.write("Conversion led to invalid data!\n")
    return lib.dump(fname.encode('ascii'), ctypes.c_ulonglong(offset * ctypes.sizeof(Db)), d, ctypes.c_ulonglong(len(db) * ctypes.sizeof(Db)), 1 if erase else 0) // ctypes.sizeof(Db)

def load_db(fname, start=0, end=0):
    if(end - start <= 0):
        import sys
        sys.stderr.write("!!! WARN: SIZE TO READ IS 0!! ({} {} {})\n".format(fname, start, end))
    c_data = lib.load(fname.encode('ascii'), start * ctypes.sizeof(Db), (end - start) * ctypes.sizeof(Db), 0)
    db_ptr = ctypes.cast(c_data.data, ctypes.POINTER(Db))
    try:
        for x in range(int(c_data.n / ctypes.sizeof(Db))):
            xxxxxx = db_ptr[x].sequence.decode('ascii')
    except:
        import sys
        sys.stderr.write("During load_db: after load {} (-> {}) {} (-> {} -> {}): {} (= {}) in {}: FAIL\n".format(start, start * ctypes.sizeof(Db), end, (end - start), (end-start)*ctypes.sizeof(Db), c_data.n, c_data.n / ctypes.sizeof(Db), fname))
    ret = c_data_to_npy(c_data, Db, blackboard.DB_DTYPE)
    return ret

def dump_res(fname, res, offset=0):
    r = npy_data_to_c(res, Res)
    return lib.dump(fname.encode('ascii'), ctypes.c_ulonglong(offset * ctypes.sizeof(Res)), r, ctypes.c_ulonglong(len(res) * ctypes.sizeof(Res)), 1) // ctypes.sizeof(Res)

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
    return ret

def dump_seq(fname, data, offset=0, erase=True):
    r = seq_to_c(data)
    return lib.dump(fname.encode('ascii'), ctypes.c_ulonglong(offset * (ctypes.sizeof(ctypes.c_float) + 1) * 128), r, ctypes.c_ulonglong(len(data) * (ctypes.sizeof(ctypes.c_float) + 1) * 128), 1 if erase else 0) // ((ctypes.sizeof(ctypes.c_float) + 1) * 128)

def load_res(fname, start=0, end=0):
    c_data = lib.load(fname.encode('ascii'), start * ctypes.sizeof(Res), (end - start) * ctypes.sizeof(Res), 0)
    ret = c_data_to_npy(c_data, Res, blackboard.RES_DTYPE)
    return ret

def sort(fnames, out, klass):
    flist = (ctypes.c_char_p * len(fnames))()
    for i in range(len(fnames)):
        flist[i] = ctypes.c_char_p(fnames[i].encode('ascii'))
    out_size = lib.merge_sort(len(fnames), flist, out.encode('ascii'), ctypes.sizeof(klass), blackboard.config['performance'].getint('sort nodes'), blackboard.config['performance'].getint('merge batch size'), blackboard.config['performance'].getint('merge nodes'))
    return out_size

def find(fname, klass, low, high):
    import sys
    sys.stderr.write("Find: {} ({}-{})\n".format(fname, low, high))
    ret = lib.find_data(fname.encode('ascii'), blackboard.config['performance'].getint('search batch size'), ctypes.sizeof(klass), ctypes.c_float(low), ctypes.c_float(high))
    sys.stderr.write("Found: {} ({}-{})\n".format(fname, low, high))
    return ret.start, ret.end
