import numpy
import time
import pickle

if __package__ is None or __package__ == '':
    import blackboard
else:
    from . import blackboard

# This is an example user script containing user functions that may be specified in the config

model = None
def predict_length(queries):
    batch_size = blackboard.config['postprocessing.length'].getint('batch size')
    device = blackboard.config['postprocessing.length']['device']
    import torch

    if __package__ is None or __package__ == '':
        from ml import length_comb_plus as length_model
    else:
        from .ml import length_comb_plus as length_model

    global model
    if model is None:      
        model = length_model.Model()
        model.to(device)
        model.load_state_dict(torch.load(blackboard.here('ml/best_nice_redo_ft1.pkl'), map_location=device))

    ret = []
    batch = []
    for iq, query in enumerate(queries):
        spec = query['spec'].data
        spec = spec[:length_model.PROT_TGT_LEN]
        precmass = query['mass']

        spec_raw = numpy.zeros((length_model.PROT_TGT_LEN,), dtype='float32')
        for mz, intens in spec:
            if mz == 0:
                break
            if mz / length_model.SIZE_RESOLUTION_FACTOR >= length_model.PROT_TGT_LEN - 0.5:
                break
            spec_raw[int(numpy.round(mz / length_model.SIZE_RESOLUTION_FACTOR))] += intens
        max = spec_raw.max()
        if max != 0:
            spec_raw /= max

        extra = query['meta'].data

        batch.append([spec_raw, precmass, extra])
        if len(batch) % batch_size == 0 or (len(batch) > 0 and iq == len(queries)-1):
            spec_batch = numpy.array([b[0] for b in batch])
            precmass_batch = numpy.array([b[1] for b in batch]).reshape((-1, 1)) / 2000.
            out = model(torch.FloatTensor(spec_batch).to(device), torch.FloatTensor(precmass_batch).to(device))
            preds = numpy.exp(out['comb'].view(-1, length_model.GT_MAX_LGT-length_model.GT_MIN_LGT+1).detach().cpu().numpy())
            for ib in range(len(batch)):
                if batch[ib][-1] is not None:
                    ret.append({'LgtPred': preds[ib], **batch[ib][-1]})
                else:
                    ret.append({'LgtPred': preds[ib]})
            batch = []
    return ret

def postprocess_for_length(start, end):
    import glob
    import os
    import sqlite3

    if __package__ is None or __package__ == '':
        from ml import length_reg_pt as length_model
    else:
        from .ml import length_reg_pt as length_model

    fname_pattern = list(filter(lambda x: len(x) > 0, blackboard.config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_*_pepidpart.sqlite"
    fname_path = os.path.join(blackboard.TMP_PATH, fname_pattern)

    files = sorted(glob.glob(fname_path))[start:end]
    if len(files) == 0:
        return

    meta = [f.replace("pepidpart.sqlite", "pepidpart_meta.sqlite") for f in files]

    header = blackboard.RES_COLS

    queries_file = blackboard.DB_PATH + "_q.sqlite"

    for fi in range(len(files)):
        while True:
            try:
                conn = sqlite3.connect("file:{}?cache=shared".format(files[fi]), detect_types=1, uri=True, timeout=0.1)
                conn_meta = sqlite3.connect("file:{}?cache=shared".format(meta[fi]), detect_types=1, uri=True, timeout=0.1)
                conn_query = sqlite3.connect("file:{}?cache=shared".format(queries_file), detect_types=1, uri=True, timeout=0.1)
                conn.row_factory = sqlite3.Row
                conn_meta.row_factory = sqlite3.Row
                conn_query.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur_meta = conn_meta.cursor()
                cur_query = conn_query.cursor()
                blackboard.execute(cur, "PRAGMA synchronous=OFF;")
                blackboard.execute(cur_meta, "PRAGMA synchronous=OFF;")
                blackboard.execute(cur_query, "PRAGMA synchronous=OFF;")
                blackboard.execute(cur, "PRAGMA temp_store_directory='{}';".format(blackboard.config['data']['tmpdir']))
                blackboard.execute(cur_meta, "PRAGMA temp_store_directory='{}';".format(blackboard.config['data']['tmpdir']))
                blackboard.execute(cur_query, "PRAGMA temp_store_directory='{}';".format(blackboard.config['data']['tmpdir']))
                break
            except:
                time.sleep(1)
                continue

        blackboard.execute(cur, "SELECT qrow, rrow, seq, score FROM results;")
        fetch_batch_size = 62000 # The maximum batch size supported by the default sqlite engine is a bit more than 62000

        while True:
            results_base = cur.fetchmany(fetch_batch_size)
            results_base = [dict(r) for r in results_base]

            if len(results_base) == 0:
                break

            blackboard.execute(cur_meta, "SELECT rrow, data FROM meta WHERE rrow IN ({}) ORDER BY rrow ASC;".format(",".join([str(r['rrow']) for r in results_base])))
            blackboard.execute(cur_query, "SELECT rowid, meta FROM queries WHERE rowid IN ({}) ORDER BY rowid ASC;".format(",".join(numpy.unique([str(r['qrow']) for r in results_base]))))

            results = cur_meta.fetchall()
            qmeta = cur_query.fetchall()
            qmeta = [dict(qm) for qm in qmeta]
            qmeta = {qm['rowid'] : qm['meta'] for qm in qmeta}

            new_meta = []
            for idata, data in enumerate(results):
                data = dict(data)
                cand_lgt = len(results_base[idata]['seq'])
                m = eval(data['data'])
                preds = numpy.asarray(qmeta[results_base[idata]['qrow']].data['LgtPred'])
                m['LgtPred'] = preds.argmax(axis=-1) + length_model.GT_MIN_LGT
                m['LgtProb'] = preds[cand_lgt - length_model.GT_MIN_LGT] if (length_model.GT_MIN_LGT <= cand_lgt <= length_model.GT_MAX_LGT) else 0
                m['LgtRelProb'] = m['LgtProb'] / preds.max(axis=-1)
                m['LgtDelta'] = m['LgtPred'] - cand_lgt
                m['LgtDeltaAbs'] = abs(m['LgtPred'] - cand_lgt)
                m['LgtScore'] = m['LgtProb'] * results_base[idata]['score']
                m['LgtRelScore'] = m['LgtRelProb'] * results_base[idata]['score']
                new_meta.append({'rrow': data['rrow'], 'data': str(m)})
            blackboard.executemany(cur_meta, 'UPDATE meta SET data = :data WHERE rrow = :rrow;', new_meta)
            conn_meta.commit()

        del cur
        del conn
        del cur_meta
        del conn_meta
        del cur_query
        del conn_query

def length_filter(cands, query):
    meta = query['meta'].data
    best_lgt = meta['LgtPred'].argmax(axis=-1) + 6
    ret = []
    for c in cands:
        if best_lgt-1 <= c['length'] <= best_lgt+1:
            ret.append(c)
    return ret

def stub(*args):
    return None
