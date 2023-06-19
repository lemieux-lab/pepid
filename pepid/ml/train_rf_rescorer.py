import sys
import numpy
import math
import pickle
import msgpack
import tqdm
import os
import sqlite3
import struct

if __package__ is None or __package__ == '':
    from pepid import blackboard
    from pepid import pepid_utils
    from pepid import queries
    from pepid import pepid_mp
else:
    from .. import blackboard
    from .. import pepid_utils
    from .. import queries
    from .. import pepid_mp

def run(cfg_file):
    log_level = blackboard.config['logging']['level'].lower()
    if blackboard.config['misc.tsv_to_pin'].getboolean('enabled'):
        nworkers = blackboard.config['rescoring.finetune_rf'].getint('pin workers')
        batch_size = blackboard.config['rescoring.finetune_rf'].getint('pin batch size')
        n_total = queries.count_queries()
        n_batches = math.ceil(n_total / batch_size)
        spec = [(blackboard.here("pin_node.py"), nworkers, n_batches,
                        [struct.pack("!cI{}sc".format(len(blackboard.TMP_PATH)), bytes([0x00]), len(blackboard.TMP_PATH), blackboard.TMP_PATH.encode("utf-8"), "$".encode("utf-8")) for _ in range(nworkers)],
                        [struct.pack("!cQQc", bytes([0x01]), b * batch_size, min((b+1) * batch_size, n_total), "$".encode("utf-8")) for b in range(n_batches)],
                        [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(nworkers)])]

        pepid_mp.handle_nodes("PIN Generation", spec, cfg_file=cfg_file, disable=log_level in ['fatal', 'error', 'warning'])

    in_fname = blackboard.config['data']['output']
    fname, fext = in_fname.rsplit('.', 1)
    suffix = blackboard.config['rescoring']['suffix']
    pin_fname = fname + suffix + "_pin.tsv"

    f = open(pin_fname)
    pin_header = next(f).strip().split('\t')
    
    import glob

    data = []

    conn = sqlite3.connect(blackboard.DB_PATH + "_q.sqlite", detect_types=1)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    forbidden_feats = set(['PSMId', 'Proteins', 'Peptide', 'ScanNr', 'Label'])
    length_feats = set(['LgtDelta', 'LgtDeltaAbs', 'LgtPred', 'LgtProb', 'LgtProbDeltaBest', 'LgtProbDeltaNext', 'LgtProbDeltaPrev', 'LgtProbDeltaWorst', 'LgtRelProb', 'LgtRelScore', 'LgtScore'])
    specgen_feats = set(['MLCorr'])

    # XXX: To disable length and/or specgen feats to generate the partial rescorers:
    #forbidden_feats |= length_feats
    #forbidden_feats |= specgen_feats

    the_feats = [f for f in pin_header if f not in forbidden_feats]
    the_feats = numpy.unique(the_feats).tolist()

    staged = []
    prev_title = None
    for li, l in tqdm.tqdm(enumerate(f), desc="reading from pin...", disable=(log_level in ['fatal', 'error', 'warning'])):
        fields = l.strip().split("\t")
        score = float(fields[pin_header.index('score')])
        title = fields[pin_header.index('PSMId')]
        if title != prev_title:
            prev_title = title
            if len(staged) > 0:
                data.extend(staged)
                del staged
                staged = []
        if not math.isinf(score):
            staged.append((title, fields[pin_header.index("Peptide")][2:-2], score, *[float(fields[pin_header.index(f)]) for f in the_feats], float(fields[pin_header.index('Label')]), -1))

    data.extend(staged)
    del staged

    data = numpy.asarray(data, dtype=[('title', object), ('seq', object), ('s', 'float32'), *[(feat, 'float32') for feat in the_feats], ('target', 'float32'), ('gt', 'float32')])

    bs = 100000
    meta = {}
    for i in tqdm.tqdm(range(0, len(data), bs), desc='collecting metadata', total = numpy.ceil(len(data) / bs), disable=(log_level in ['warning', 'error', 'fatal'])):
        cur.execute("SELECT title, meta FROM queries WHERE title IN ({});".format(",".join(["\"{}\"".format(t) for t in data['title'][i:i+bs]])))
        things = cur.fetchall()
        meta = meta | {m['title'] : pickle.loads(m['meta'])['mgf:SEQ'] for m in things}

    for di in tqdm.tqdm(range(len(data)), desc='adding gt info', total = len(data), disable=(log_level in ['warning', 'error', 'fatal'])):
        seq = meta[data[di]['title']]
        data[di]['gt'] = ((seq.replace("M(ox)", "M[15.994915]").replace("C", "C[57.0215]") == data[di]['seq']) - 0.5)*2

    cur.close()
    conn.close()

    numpy.random.shuffle(data)
    every = data

    names = every.dtype.names
    feat_idxs = [names.index(feat) for feat in the_feats]

    import sklearn
    from sklearn import preprocessing
    from sklearn import ensemble
    from sklearn import model_selection

    blackboard.LOG.info("Fitting...")

    best_dir = True # i.e. bigger = better
    best_feat = 's'
    best_score = 0

    feat_ipt = blackboard.config['rescoring.finetune_rf']['score']
    if feat_ipt == '':
        feat_ipt = None
    feat_dir = blackboard.config['rescoring.finetune_rf'].getboolean('descending')

    if feat_ipt is None:
        for feat in the_feats + ['s']:
            for direction in [False, True]:
                every.sort(order=[feat, 'target'])
                if direction:
                    every = every[::-1]

                fdrs = numpy.asarray(pepid_utils.calc_qval(every[feat], every['target'] > 0))
                aw = numpy.argwhere(fdrs <= 0.01)
                min_s = every[feat][(aw.reshape((-1,))[-1])+1] if len(aw) > 0 else (every[feat].max()+1) if direction else (every[feat].min()-1)

                score = (every[feat] > min_s).sum() if direction else (every[feat] < min_s).sum()
                if score > best_score:
                    best_dir = direction
                    best_feat = feat

                blackboard.LOG.debug("[{}|{}] Identified past 1% FDR: {}".format(feat, "+" if direction else "-", score))
    else:
        best_feat = feat_ipt
        best_dir = feat_dir

    feat = best_feat
    direction = best_dir

    if not direction:
        every[feat] = -every[feat]

    every.sort(order=[feat, 'target'])
    every = every[::-1]

    fdrs = numpy.asarray(pepid_utils.calc_qval(every[feat], every['target'] > 0))
    aw = numpy.argwhere(fdrs <= 0.01)
    min_s = every[feat][aw.reshape((-1,))[-1]+1] if len(aw) > 0 else every[feat].max()+1

    keys = numpy.unique(every['title'])
    grouped_data = {k: [] for k in keys}
    for d in every:
        if len(grouped_data[d['title']]) >= 1:
            continue
        grouped_data[d['title']].append(d)
    data = numpy.hstack([grouped_data[group] for group in grouped_data])

    data.sort(order=[feat, 'target'])
    data = data[::-1]

    fdrs = numpy.asarray(pepid_utils.calc_qval(data[feat], data['target'] > 0))
    aw = numpy.argwhere(fdrs <= 0.01)
    min_st = data[feat][aw.reshape((-1,))[-1]+1] if len(aw) > 0 else data[feat].max()+1

    fdrs = numpy.asarray(pepid_utils.calc_qval(data[feat], data['gt'] > 0))
    aw = numpy.argwhere(fdrs <= 0.01)
    min_sgt = data[feat][aw.reshape((-1,))[-1]+1] if len(aw) > 0 else data[feat].max()+1

    blackboard.LOG.info("Total: {}".format(len(every)))
    blackboard.LOG.info("[{}] Identified past 1% FDR: {}".format(feat, (every[feat] > min_s).sum()))
    blackboard.LOG.info("Top-1: [{}] Identified past 1% FDR: {}".format(feat, (data[feat] > min_st).sum()))
    blackboard.LOG.info("Top-1: [{}] Identified past 1% FDP: {}".format(feat, (data[feat] > min_sgt).sum()))

    fdrs = numpy.asarray(pepid_utils.calc_qval(every[feat], every['target'] > 0))
    aw = numpy.argwhere(fdrs <= 0.01)
    min_s = every[feat][aw.reshape((-1,))[-1]+1] if len(aw) > 0 else every[feat].max()+1

    every_groups = []
    prev_title = None
    idx = -1
    for title in every['title']:
        if prev_title != title:
            prev_title = title
            idx += 1
        every_groups.append(idx) 

    is_index = numpy.asarray([(t['target'] < -0.5 or (t[feat] > min_s)) for t in every])
    every_feats = numpy.asarray([t[the_feats] for t in every]).view('float32').reshape((-1, len(the_feats)))
    every_labs = numpy.asarray([1 if t['target'] > 0.5 else -1 for t in every])

    n_folds = 10
    fold = model_selection.GroupKFold(n_splits=n_folds)

    best_model = None
    best_idx = -1

    for i, (train_idxs, test_idxs) in enumerate(fold.split(every, groups=every_groups)):
        blackboard.LOG.debug("Fold {}: Init data".format(i+1))

        train_feats = every_feats[train_idxs][is_index[train_idxs]]
        train_labs = every_labs[train_idxs][is_index[train_idxs]]

        blackboard.LOG.debug("Fold {}: Train".format(i+1))
        rf = ensemble.RandomForestClassifier(n_jobs=8)
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_feats)

        rf.fit(scaler.transform(every_feats[train_idxs]), [1 if t['gt'] > 0.5 else 0 for t in every[train_idxs]])

        blackboard.LOG.debug("Fold {}: Test".format(i+1))
        test_feats = every_feats[test_idxs]
        test_out = numpy.asarray(rf.predict_proba(scaler.transform(test_feats))[:,1])

        fdrs = numpy.asarray(pepid_utils.calc_qval(test_out, every[test_idxs]['gt'] > 0.5))
        aw = numpy.argwhere(fdrs <= 0.01)
        min_idx = aw.reshape((-1,))[-1]

        if min_idx > best_idx:
            best_model = rf

        blackboard.LOG.debug("Fold {}: Done".format(i+1))

    blackboard.LOG.debug("Done")

    # we save the feats in order so we can correctly reorder feats and check if they're really present at load time... while the rescorer is very specific, this should give slightly more flexibility.
    pickle.dump({'model': best_model, 'scaler': scaler, 'feats': the_feats}, open(blackboard.here("ml/rescorer_rf.pkl"), "wb"))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        blackboard.LOG.error("USAGE: {} config.cfg\n".format(sys.argv[0]))
        sys.exit(-1)

    blackboard.config.read(blackboard.here("data/default.cfg"))
    blackboard.config.read(sys.argv[1])

    blackboard.setup_constants()

    run(sys.argv[1])
