import numpy
import time
import pickle
#import ujson as deser
#import pickle as deser
#import quickle as deser

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

        batch.append([spec_raw, precmass])
        if (len(batch) % batch_size == 0) or (iq == len(queries)-1):
            spec_batch = numpy.array([b[0] for b in batch])
            precmass_batch = numpy.array([b[1] for b in batch]).reshape((-1, 1)) / 2000.
            out = model(torch.FloatTensor(spec_batch).to(device), torch.FloatTensor(precmass_batch).to(device))
            preds = numpy.exp(out['comb'].view(-1, length_model.GT_MAX_LGT-length_model.GT_MIN_LGT+1).detach().cpu().numpy())
            for ib in range(len(batch)):
                ret.append({'META_LgtPred': pickle.dumps(preds[ib])})
                #ret.append({'META_LgtPred': preds[ib]})
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

    header = blackboard.RES_COLS

    queries_file = blackboard.DB_PATH + "_q.sqlite"

    meta_cols = []
    extension_feats = set(['META_LgtProb', 'META_LgtRelProb', 'META_LgtPred', 'META_LgtDelta', 'META_LgtDeltaAbs',  'META_LgtProbDeltaBest', 'META_LgtProbDeltaWorst', 'META_LgtProbDeltaNext', 'META_LgtProbDeltaPrev'])

    for fi in range(len(files)):
        conn = sqlite3.connect("file:{}?cache=shared".format(files[fi]), detect_types=1, uri=True, timeout=0.1)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        mod_cur = conn.cursor()
        blackboard.execute(cur, "PRAGMA synchronous=OFF;")
        blackboard.execute(cur, "PRAGMA temp_store_directory='{}';".format(blackboard.config['data']['tmpdir']))

        if len(meta_cols) == 0:
            blackboard.execute(cur, "SELECT * FROM results LIMIT 1;")
            meta_cols = list(dict(cur.fetchone()).keys())
            for feat in extension_feats:
                if feat not in meta_cols:
                    cur.execute("ALTER TABLE results ADD COLUMN {} REAL;".format(feat))
            conn.commit()

        blackboard.execute(cur, "CREATE INDEX IF NOT EXISTS qrow_index ON results (qrow);")
        blackboard.execute(cur, "ATTACH DATABASE ? AS queries;", (queries_file,)) 
        blackboard.execute(cur, "SELECT r.rowid, r.seq, q.META_LgtPred FROM (SELECT rowid, qrow, seq FROM results) r INNER JOIN (SELECT rowid, META_LgtPred FROM queries) q ON q.rowid = r.qrow ORDER BY r.qrow;")
        fetch_batch_size = 62000 # The maximum batch size supported by the default sqlite engine is a bit more than 62000

        while True:
            results_base = cur.fetchmany(fetch_batch_size)
            results_base = [dict(r) for r in results_base]

            if len(results_base) == 0:
                break

            new_meta = []
            for idata, data in enumerate(results_base):
                m = {}

                cand_lgt = len(data['seq'])

                preds = pickle.loads(data['META_LgtPred'])
                #preds = numpy.frombuffer(q[data['qrow']]['META_LgtPred'], dtype='float32')
                bests = numpy.argsort(preds, axis=-1)[::-1]
                m['META_LgtPred'] = float(preds.argmax(axis=-1) + length_model.GT_MIN_LGT)
                lgt_prob = float(preds[cand_lgt - length_model.GT_MIN_LGT] if (length_model.GT_MIN_LGT <= cand_lgt <= length_model.GT_MAX_LGT) else 0)

                m['META_LgtProb'] = float(preds[cand_lgt - length_model.GT_MIN_LGT] if (length_model.GT_MIN_LGT <= cand_lgt <= length_model.GT_MAX_LGT) else 0)
                m['META_LgtRelProb'] = float(m['META_LgtProb'] / preds.max(axis=-1))
                m['META_LgtProbDeltaBest'] = float(m['META_LgtProb'] - preds[bests[0]])
                m['META_LgtProbDeltaWorst'] = float(m['META_LgtProb'] - preds[bests[-1]])
                m['META_LgtProbDeltaPrev'] = float(0 if m['META_LgtProb'] == 0 else (m['META_LgtProb'] - preds[max(preds[bests].tolist().index(preds[cand_lgt - length_model.GT_MIN_LGT]) - 1, 0)]))
                m['META_LgtProbDeltaNext'] = float(0 if m['META_LgtProb'] == 0 else (m['META_LgtProb'] - preds[min(preds[bests].tolist().index(preds[cand_lgt - length_model.GT_MIN_LGT]) + 1, len(preds)-1)]))
                #m['META_LgtScoreModel'] = float(1 / (1 + numpy.exp(-0.05 * (data['score'] + m['META_LgtRelProb']))))
                m['META_LgtDelta'] = float(m['META_LgtPred'] - cand_lgt)
                m['META_LgtDeltaAbs'] = float(abs(preds.argmax(axis=-1) + length_model.GT_MIN_LGT - cand_lgt))
                #m['LgtScore'] = m['LgtProb'] * results_base[idata]['score']
                #m['LgtRelScore'] = m['LgtRelProb'] * results_base[idata]['score']
                new_meta.append({'rowid': data['rowid'], **m})
            blackboard.executemany(mod_cur, 'UPDATE results SET {} WHERE rowid = :rowid;'.format(",".join(["{} = :{}".format(k, k) for k in new_meta[-1].keys() if k != 'rowid'])), new_meta)
            conn.commit()

        del cur
        del mod_cur
        del conn
        meta_cols = []

def length_filter(cands, query):
    meta = deser.loads(query['meta'])
    best_lgt = meta['META_LgtPred'].argmax(axis=-1) + 6
    ret = []
    for c in cands:
        if best_lgt-1 <= c['length'] <= best_lgt+1:
            ret.append(c)
    return ret

def stub(*args):
    return None
