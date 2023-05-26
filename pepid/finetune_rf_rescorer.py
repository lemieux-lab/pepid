import sys
import numpy
import math
import pickle
import msgpack
import tqdm
import struct

if __package__ is None or __package__ == '':
    import blackboard
    import pepid_utils
    import queries
    import pepid_mp
else:
    from . import blackboard
    from . import pepid_utils
    from . import queries
    from . import pepid_mp

def run(cfg):
    blackboard.config.read(blackboard.here("data/default.cfg"))
    blackboard.config.read(cfg)

    blackboard.setup_constants()

    log_level = blackboard.config['logging']['level'].lower()
    if blackboard.config['misc.tsv_to_pin'].getboolean('enabled'):
        in_fname = blackboard.config['data']['output']
        fname, fext = in_fname.rsplit('.', 1)
        suffix = blackboard.config['rescoring']['suffix']
        pin_name = fname + suffix + "_pin.tsv"
 
        inf = open(in_fname, 'r')
        header = next(inf).strip().split("\t")
        line = next(inf).strip().split('\t')
        inf.close()
        pinf = open(pin_name, 'w')
        pinf.write("\t".join(pepid_utils.generate_pin_header(header, line)) + "\n")
        pinf.close()

        nworkers = blackboard.config['rescoring.finetune_rf'].getint('pin workers')
        batch_size = blackboard.config['rescoring.finetune_rf'].getint('pin batch size')
        n_total = queries.count_queries()
        n_batches = math.ceil(n_total / batch_size)
        spec = [(blackboard.here("pin_node.py"), nworkers, n_batches,
                        [struct.pack("!cI{}sc".format(len(blackboard.TMP_PATH)), bytes([0x00]), len(blackboard.TMP_PATH), blackboard.TMP_PATH.encode("utf-8"), "$".encode("utf-8")) for _ in range(nworkers)],
                        [struct.pack("!cQQc", bytes([0x01]), b * batch_size, min((b+1) * batch_size, n_total), "$".encode("utf-8")) for b in range(n_batches)],
                        [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(nworkers)])]

        pepid_mp.handle_nodes("PIN Generation", spec, cfg_file=cfg, tqdm_silence=log_level in ['fatal', 'error', 'warning'])

    in_fname = blackboard.config['data']['output']
    fname, fext = in_fname.rsplit('.', 1)
    suffix = blackboard.config['rescoring']['suffix']
    pin_fname = fname + suffix + "_pin.tsv"

    model_path = blackboard.config['rescoring.finetune_rf']['model']
    preprocessor_path = blackboard.config['rescoring.finetune_rf']['preprocessor']

    model = pickle.load(open(blackboard.here(model_path), "rb"))
    scaler = pickle.load(open(blackboard.here(preprocessor_path), "rb"))

    f = open(pin_fname, 'r')
    header = next(f).strip().split("\t")
    feats = header

    f.close()

    template = blackboard.pin_template()
    feats = [x for x in feats if x not in template]

    import glob

    data = []

    import sqlite3
    connq = sqlite3.connect("/scratch/zumerj/tmp/pepidrun_yeast/human_q.sqlite", detect_types=1)
    connc = sqlite3.connect("/scratch/zumerj/tmp/pepidrun_yeast/human_cands.sqlite", detect_types=1)
    connq.row_factory = sqlite3.Row
    connc.row_factory = sqlite3.Row
    curq = connq.cursor()
    curc = connc.cursor()

    f = open(pin_fname, 'r')
    header = next(f).strip().split("\t")
    for li, l in enumerate(f):
        fields = l.strip().split("\t")
        score = float(fields[header.index('score')])
        title = fields[header.index('PSMId')]
        if not math.isinf(score):
            data.append((title, fields[header.index("Peptide")][2:-2], score, *[float(fields[header.index(f)]) for f in feats], float(fields[header.index('Label')])))

    data = numpy.asarray(data, dtype=[('title', object), ('seq', object), ('s', 'float32'), *[(feat, 'float32') for feat in feats], ('target', 'float32')])

    bs = 100000

    numpy.random.shuffle(data)
    every = data

    names = every.dtype.names
    feat_idxs = [names.index(feat) for feat in feats]

    import sklearn
    from sklearn import ensemble
    from sklearn import preprocessing
    from sklearn import model_selection

    blackboard.LOG.info("Fitting...")

    best_dir = True # i.e. bigger = better
    best_feat = 's'
    best_score = 0

    feat_ipt = blackboard.config['rescoring.finetune_rf']['score']
    feat_dir = blackboard.config['rescoring.finetune_rf'].getboolean('descending')

    if feat_ipt is None:
        for feat in feats + ['s']:
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

                blackboard.LOG.debug("[{}|{}] Identified past 1% FDR: {}".format(feat, ("+" if direction else "-"), score))
    else:
        best_feat = feat_ipt
        direction = feat_dir

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

    blackboard.LOG.info("Total: {}".format(len(every)))
    blackboard.LOG.info("[{}] Identified past 1% FDR: {}".format(feat, (every[feat] > min_s).sum()))
    blackboard.LOG.info("Top-1: [{}] Identified past 1% FDR: {}".format(feat, (data[feat] > min_st).sum()))

    out_transformed = numpy.zeros((every.shape[0],), dtype='float32')

    fdrs = numpy.asarray(pepid_utils.calc_qval(every[feat], every['target'] > 0))
    aw = numpy.argwhere(fdrs <= 0.01)
    min_s = every[feat][aw.reshape((-1,))[-1]+1] if len(aw) > 0 else every[feat].max()+1

    lvl = every[int(0.5 * len(every))][feat]

    every_groups = []
    prev_title = None
    idx = -1
    for title in every['title']:
        if prev_title != title:
            prev_title = title
            idx += 1
        every_groups.append(idx) 

    is_index = numpy.asarray([(t['target'] < -0.5 or (t[feat] > min_s)) for t in every])
    every_feats = numpy.asarray([t[feats] for t in every]).view('float32').reshape((-1, len(feats)))
    every_labs = numpy.asarray([1 if t['target'] > 0.5 else -1 for t in every])

    n_folds = 10
    fold = model_selection.GroupKFold(n_splits=n_folds)

    for i, (train_idxs, test_idxs) in enumerate(fold.split(every, groups=every_groups)):
        blackboard.LOG.debug("Fold {}: Init data".format(i+1))

        train_feats = every_feats[train_idxs] #[is_index[train_idxs]]
        train_labs = every_labs[train_idxs] #[is_index[train_idxs]]

        blackboard.LOG.debug("Fold {}: Train".format(i+1))
        rf_gt = model
        rf_finetune = ensemble.RandomForestClassifier(n_jobs=-1)
        scaler = scaler

        train_labs = [1 if (t[feat] >= lvl and t['target'] > 0.5) else 0 for t in every[train_idxs]]

        rf_finetune.fit(scaler.transform(every_feats[train_idxs]), train_labs)

        blackboard.LOG.debug("Fold {}: Test".format(i+1))

        test_feats = every_feats[test_idxs]
        old = rf_gt.predict_proba(scaler.transform(test_feats))[:,1]
        new = rf_finetune.predict_proba(scaler.transform(test_feats))[:,1]
        test_out = numpy.asarray(0.75*old + 0.25*new)

        out_transformed[test_idxs] = test_out

        blackboard.LOG.debug("Fold {}: Done".format(i+1))

    blackboard.LOG.debug("Done")

    keys = numpy.unique(every['title'])
    grouped_data = {k: [] for k in keys}
    data = []
    for i in range(len(every)):
        grouped_data[every[i]['title']].append((out_transformed[i], every[i]['title'], every[i]['seq'], every[i]['target']))
    for g in grouped_data:
        data.append(grouped_data[g][numpy.argmax([gg[0] for gg in grouped_data[g]])])
    data = numpy.asarray(data, dtype=[('s', numpy.float32), ('title', object), ('seq', object), ('target', numpy.int32)])
    data.sort(order=['s', 'target'])
    data = data[::-1]

    fdrs = numpy.asarray(pepid_utils.calc_qval(data['s'], data['target'] > 0))
    aw = numpy.argwhere(fdrs <= 0.01)
    min_st = data['s'][min(len(data)-1, aw.reshape((-1,))[-1]+1)] if len(aw) > 0 else data['s'].max()+1
    blackboard.LOG.debug("FDRS/target: {} {}".format(fdrs[0], fdrs[-1]))

    blackboard.LOG.info("Identified past 1% FDR: {}".format((data['s'] > min_st).sum()))

    fin = open(in_fname, "r")
    assoc = {}
    in_header = next(fin).strip().split("\t")
    for l in fin:
        il = l.strip().split("\t")
        assoc[(il[in_header.index('title')], il[in_header.index('modseq')])] = il
    fin.close()
    for d in data:
        fields = assoc[(d['title'], d['seq'])]
        fields[in_header.index('score')] = numpy.format_float_positional(d['s'], trim='0', precision=12)
        yield fields

if __name__ == '__main__':
    if len(sys.argv) != 2:
        blackboard.LOG.error("USAGE: {} config.cfg\n".format(sys.argv[0]))
        sys.exit(-1)

    blackboard.config.read(blackboard.here("data/default.cfg"))
    blackboard.config.read(sys.argv[1])

    blackboard.setup_constants()

    in_fname = blackboard.config['data']['output']
    fname, fext = in_fname.rsplit('.', 1)
    out_fname = fname + blackboard.config['rescoring']['suffix'] + "." + fext
    outf = open(out_fname, 'w')
    outf.write(next(open(in_fname, 'r')))

    for l in run(sys.argv[1]):
        outf.write(l)
        
    outf.close()
