import numpy
import time
import msgpack
import pickle
import sys

if __package__ is None or __package__ == '':
    import blackboard
    import pepid_utils
else:
    from . import blackboard
    from . import pepid_utils

import numba

@numba.njit(locals={'max_mz': numba.int32, 'mult': numba.float32})
def correlate_spectra(blit, mlspec, max_mz, mult):
    corr = 0
    sqr_ml = 0
    sqr_blit = 0
    for mz, intens in mlspec:
        if int(mz) >= max_mz / mult:
            break
        blit_int = blit[int(mz)]
        corr += blit_int * intens
        sqr_ml += intens**2
        sqr_blit += blit_int**2
    corr /= (numpy.sqrt(sqr_blit) * numpy.sqrt(sqr_ml) + 1e-10)

    return corr

def specgen_features(header, lines):
    import sqlite3

    batch_size = blackboard.config['pin.specgen_features'].getint('batch size')

    cand_file = blackboard.DB_PATH + "_cands.sqlite"
    qfile = blackboard.DB_PATH + "_q.sqlite"

    if __package__ is None or __package__ == '':
        from ml import specgen
    else:
        from .ml import specgen

    conn = sqlite3.connect(cand_file, detect_types=1)
    connq = sqlite3.connect(qfile, detect_types=1)
    conn.row_factory = sqlite3.Row
    connq.row_factory = sqlite3.Row
    cur = conn.cursor()
    curq = connq.cursor()

    ret = []

    for i in range(0, len(lines), batch_size):
        cands = [l[header.index('candrow')] for ll in lines[i:i+batch_size] for l in ll]
        quers = [l[header.index('qrow')] for ll in lines[i:i+batch_size] for l in ll]
        cur.execute("SELECT rowid, meta FROM candidates WHERE rowid IN ({}) ORDER BY rowid;".format(",".join(cands)))
        curq.execute("SELECT rowid, spec FROM queries WHERE rowid in ({}) ORDER BY rowid;".format(",".join(quers)))
        allcands = cur.fetchall()
        extras = {r['rowid'] : pickle.loads(r['meta'])['MLSpec'] for r in allcands}
        specs = {r['rowid'] : pickle.loads(r['spec']) for r in curq.fetchall()}

        for query_lines in lines[i:i+batch_size]:
            ret.append([])
            for line in query_lines:
                charge = int(line[header.index('query_charge')])-1
                spec = specs[int(line[header.index('qrow')])]
                blit = specgen.prepare_spec(spec)
                mlspec = extras[int(line[header.index('candrow')])][charge]
                corr = correlate_spectra(blit, mlspec, specgen.MAX_MZ, 1.0 / specgen.SIZE_RESOLUTION_FACTOR)
                ret[-1].append({'MLCorr': corr})
    return ret

model = None
class predict_length(object):
    required_fields = {'queries': ['spec', 'mass', 'meta']}

    def __new__(cls, queries):
        batch_size = blackboard.config['postprocessing.length'].getint('batch size')
        device = blackboard.config['postprocessing.length']['device']
        if 'cuda' in device:
            blackboard.lock()
            gpu_lock = blackboard.acquire_lock(device)

            blackboard.lock(gpu_lock)

            blackboard.unlock()

        import torch

        if __package__ is None or __package__ == '':
            from ml import best_lgt_model as length_model
        else:
            from .ml import best_lgt_model as length_model

        global model
        if model is None:      
            model = length_model.Model()
            model.to(device)
            model.load_state_dict(torch.load(blackboard.here('ml/best_lgt_model.pkl'), map_location=device))

        ret = []
        batch = []
        for iq, query in enumerate(queries):
            spec = pickle.loads(query['spec'])
            spec = numpy.asarray(spec[:length_model.PROT_TGT_LEN], dtype='float32')
            precmass = query['mass']

            spec_raw = pepid_utils.blit_spectrum(spec, length_model.PROT_TGT_LEN, length_model.SIZE_RESOLUTION_FACTOR)
            extra = pickle.loads(query['meta'])

            batch.append([spec_raw, precmass, extra])
            if len(batch) % batch_size == 0 or (len(batch) > 0 and iq == len(queries)-1):
                spec_batch = numpy.array([b[0] for b in batch])
                precmass_batch = numpy.array([b[1] for b in batch]).reshape((-1, 1)) / 2000.
                with torch.no_grad():
                    out = model(torch.FloatTensor(spec_batch).to(device), torch.FloatTensor(precmass_batch).to(device))
                    preds = numpy.exp(out['pred'].view(-1, length_model.GT_MAX_LGT-length_model.GT_MIN_LGT+1).detach().cpu().numpy())
                for ib in range(len(batch)):
                    if batch[ib][-1] is not None:
                        ret.append({**batch[ib][-1], 'LgtPred': preds[ib]})
                    else:
                        ret.append({'LgtPred': preds[ib]})
                batch = []
        if 'cuda' in device:
            import gc

            del model
            model = None
            gc.collect()

            torch.cuda.empty_cache()
            blackboard.unlock(gpu_lock)

        return ret

def postprocess_for_length(start, end):
    import glob
    import os
    import sqlite3

    if __package__ is None or __package__ == '':
        from ml import best_lgt_model as length_model
    else:
        from .ml import best_lgt_model as length_model

    fname_pattern = list(filter(lambda x: len(x) > 0, blackboard.config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_*_pepidpart.sqlite"
    fname_path = os.path.join(blackboard.TMP_PATH, fname_pattern)

    files = sorted(glob.glob(fname_path))[start:end]
    if len(files) == 0:
        return

    meta = [f.replace("pepidpart.sqlite", "pepidpart_meta.sqlite") for f in files]

    header = blackboard.RES_COLS

    queries_file = blackboard.DB_PATH + "_q.sqlite"
    conn_query = sqlite3.connect("file:{}?cache=shared".format(queries_file), detect_types=1, uri=True, timeout=0.1)
    conn_query.row_factory = sqlite3.Row
    cur_query = conn_query.cursor()
    blackboard.execute(cur_query, "PRAGMA synchronous=OFF;")
    blackboard.execute(cur_query, "PRAGMA temp_store_directory='{}';".format(blackboard.config['data']['tmpdir']))

    for fi in range(len(files)):
        while True:
            try:
                conn = sqlite3.connect("file:{}?cache=shared".format(files[fi]), detect_types=1, uri=True, timeout=0.1)
                conn_meta = sqlite3.connect("file:{}?cache=shared".format(meta[fi]), detect_types=1, uri=True, timeout=0.1)
                conn.row_factory = sqlite3.Row
                conn_meta.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur_meta = conn_meta.cursor()
                cur_meta_mod = conn_meta.cursor()
                blackboard.execute(cur, "PRAGMA synchronous=OFF;")
                blackboard.execute(cur_meta, "PRAGMA synchronous=OFF;")
                blackboard.execute(cur, "PRAGMA temp_store_directory='{}';".format(blackboard.config['data']['tmpdir']))
                blackboard.execute(cur_meta, "PRAGMA temp_store_directory='{}';".format(blackboard.config['data']['tmpdir']))
                break
            except:
                time.sleep(1)
                continue

        blackboard.execute(cur, "SELECT qrow, rrow, seq, score FROM results ORDER BY rrow ASC;")
        blackboard.execute(cur_meta, "SELECT rrow, extra FROM meta ORDER BY rrow ASC;")
        fetch_batch_size = 62000 # The maximum batch size supported by the default sqlite engine is a bit more than 62000

        while True:
            results_base = cur.fetchmany(fetch_batch_size)
            results_base = [dict(r) for r in results_base]

            if len(results_base) == 0:
                break

            blackboard.execute(cur_query, "SELECT rowid, meta FROM queries WHERE rowid IN ({}) ORDER BY rowid ASC;".format(",".join(numpy.unique([str(r['qrow']) for r in results_base]))))

            results = [dict(r) for r in cur_meta.fetchmany(fetch_batch_size)]
            qmeta = cur_query.fetchall()
            qmeta = [dict(qm) for qm in qmeta]
            qmeta = {qm['rowid'] : qm['meta'] for qm in qmeta}

            new_meta = []
            for idata, data in enumerate(results):
                cand_lgt = len(results_base[idata]['seq'])
                m = msgpack.loads(data['extra'])
                if m is None:
                    m = {}
                preds = pickle.loads(qmeta[results_base[idata]['qrow']])['LgtPred']
                #preds = numpy.asarray(msgpack.loads(qmeta[results_base[idata]['qrow']])['LgtPred'], dtype='float32')
                bests = numpy.argsort(preds, axis=-1)[::-1]
                m['LgtPred'] = float(preds.argmax(axis=-1) + length_model.GT_MIN_LGT)
                m['LgtProb'] = float(preds[cand_lgt - length_model.GT_MIN_LGT] if (length_model.GT_MIN_LGT <= cand_lgt <= length_model.GT_MAX_LGT) else 0)
                m['LgtRelProb'] = float(m['LgtProb'] / preds.max(axis=-1))
                m['LgtDelta'] = float(m['LgtPred'] - cand_lgt)
                m['LgtDeltaAbs'] = float(abs(m['LgtPred'] - cand_lgt))
                m['LgtScore'] = float(m['LgtProb'] * results_base[idata]['score'])
                m['LgtRelScore'] = float(m['LgtRelProb'] * results_base[idata]['score'])
                m['LgtProbDeltaBest'] = float(m['LgtProb'] - preds[bests[0]])
                m['LgtProbDeltaWorst'] = float(m['LgtProb'] - preds[bests[-1]])
                m['LgtProbDeltaPrev'] = float(0 if m['LgtProb'] == 0 else (m['LgtProb'] - preds[max(preds[bests].searchsorted(preds[cand_lgt - length_model.GT_MIN_LGT]) - 1, 0)]))
                m['LgtProbDeltaNext'] = float(0 if m['LgtProb'] == 0 else (m['LgtProb'] - preds[min(preds[bests].searchsorted(preds[cand_lgt - length_model.GT_MIN_LGT]) + 1, len(preds)-1)]))
                new_meta.append({'rrow': data['rrow'], 'data': msgpack.dumps(m)})
            blackboard.executemany(cur_meta_mod, 'UPDATE meta SET extra = :data WHERE rrow = :rrow;', new_meta)
            conn_meta.commit()

        del cur
        del conn
        del cur_meta
        del conn_meta

class length_filter(object):
    required_fileds = {'candidates': ['length'], 'queries': ['meta']}

    def __new__(cls, cands, query):
        meta = msgpack.loads(query['meta'])

        best_lgts = numpy.argsort(meta['LgtPred'])[::-1] + 6

        ret = []
        for c in cands:
            probs = numpy.asarray([meta['LgtPred'][b-6] for b in best_lgts])
            if (probs[c['length']-6] >= 0.25):
                if c['length'] in best_lgts[:2]:
                    ret.append(c)
            else:
                ret.append(c)
        return ret

def stub(*args):
    return None
