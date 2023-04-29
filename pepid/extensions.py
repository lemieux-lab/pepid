import numpy
import time
import msgpack

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
            preds = numpy.exp(out['pred'].view(-1, length_model.GT_MAX_LGT-length_model.GT_MIN_LGT+1).detach().cpu().numpy()).tolist()
            for ib in range(len(batch)):
                if batch[ib][-1] is not None:
                    ret.append({**batch[ib][-1], 'LgtPred': preds[ib]})
                else:
                    ret.append({'LgtPred': preds[ib]})
            batch = []
    return ret

def insert_gt_length(queries):
    ret = []
    for iq, query in enumerate(queries):
        extra = query['meta'].data
        if 'mgf:SEQ' not in extra:
            blackboard.LOG.error("Missing key 'mgf:SEQ' for insert_gt_length. Did you forget to run pepid_mgf_meta?")
            sys.exit(-2)
        else:
            lgt = len(extra['mgf:SEQ'].replace("M(ox)", "1"))
            pred = numpy.zeros((40-6+1), dtype='float32')
            pred[lgt-6] = 1.
            extra['LgtPred'] = pred.tolist()
        ret.append(extra)
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
                preds = numpy.asarray(qmeta[results_base[idata]['qrow']].data['LgtPred'])
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
                m['LgtProbDeltaPrev'] = float(0 if m['LgtProb'] == 0 else (m['LgtProb'] - preds[max(preds[bests].tolist().index(preds[cand_lgt - length_model.GT_MIN_LGT]) - 1, 0)]))
                m['LgtProbDeltaNext'] = float(0 if m['LgtProb'] == 0 else (m['LgtProb'] - preds[min(preds[bests].tolist().index(preds[cand_lgt - length_model.GT_MIN_LGT]) + 1, len(preds)-1)]))
                new_meta.append({'rrow': data['rrow'], 'data': msgpack.dumps(m)})
            blackboard.executemany(cur_meta_mod, 'UPDATE meta SET extra = :data WHERE rrow = :rrow;', new_meta)
            conn_meta.commit()

        del cur
        del conn
        del cur_meta
        del conn_meta

def length_filter(cands, query):
    meta = query['meta'].data

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
