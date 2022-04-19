import pepid_utils
import blackboard

import numpy
import tqdm

import os

import sqlite3

def load_data(f, batch_size):
    titles = {}
    lines = []

    header = []

    max_scores = int(blackboard.config['rescoring']['max scores'])

    n_titles = 0

    for il, l in enumerate(f):
        if il == 0:
            header = l.strip().split('\t')
            continue
        t = l.strip().split('\t', 1)[0]
        if t not in titles:
            if(n_titles > 0 and n_titles % batch_size == 0):
                titles = {k: numpy.array(v) for k, v in titles.items()}
                yield titles, lines
                lines = []
                titles = {}
                n_titles = 0
            n_titles += 1
            titles[t] = []
        elif len(titles[t]) >= max_scores:
            continue
        lines.append(l)
        l = l.strip().split('\t')
        #rrow = int(l[-1])
        #fname = l[-2]
        #conn = sqlite3.connect("file:" + os.path.join(blackboard.TMP_PATH, fname + "_meta.sqlite") + "?cache=shared", detect_types=1, uri=True) 
        #cur = conn.cursor()
        #cur.execute("SELECT data FROM meta WHERE rrow = ?;", (rrow,))
        #meta = eval(cur.fetchone()[0])
        #print(meta)
        qcharge = int(l[header.index('query_charge')])
        qmass = float(l[header.index('query_mass')])
        cmass = float(l[header.index('cand_mass')])
        titles[t].append(["DECOY_" in l[header.index("desc")], float(l[header.index("score")]), qcharge, qmass - cmass, qmass, cmass] + [len(l[header.index('seq')])])

    yield titles, lines

def split_data(titles):
    t = numpy.asarray(list(titles.keys()))

    all_idxs = numpy.arange(len(t)).astype('int32')
    numpy.random.shuffle(all_idxs)

    train_idxs = all_idxs[:int(0.9*len(all_idxs))]
    valid_idxs = all_idxs[int(0.9*len(all_idxs)):]

    return numpy.vstack([titles[k] for k in t[train_idxs]]), numpy.vstack([titles[k] for k in t[valid_idxs]])

def count_spectra(f):
    titles = {}
    _ = next(f)

    for l in f:
        title = l.split('\t', 1)[0]
        titles[title] = 1

    return len(titles)

def rescore(f):
    batch_size = blackboard.config['rescoring'].getint('batch size')
    n_titles = count_spectra(f)
    f.seek(0)

    import sklearn.svm
    import sklearn.ensemble
    import sklearn.linear_model
    from sklearn.metrics import roc_auc_score
    import sklearn.pipeline
    import sklearn.preprocessing

    log_level = blackboard.config['logging']['level'].lower()
    verbose_level = 2 if log_level in ['debug'] else 0

    # NOTE: this targets a total of 100 estimators if possible, with a minimum of 1 estimator per slice
    #n_est_per_fit = max(5, 100 // (n_titles // batch_size))
    #model = sklearn.ensemble.RandomForestClassifier(verbose=verbose_level, n_jobs=-1, n_estimators=0, warm_start=True)
    preproc = sklearn.preprocessing.StandardScaler()
    model = sklearn.linear_model.SGDClassifier(verbose=verbose_level, n_jobs=-1, loss='modified_huber')
    valid_set = []
 
    for titles, _ in tqdm.tqdm(load_data(f, batch_size), disable=log_level in ['fatal', 'error', 'warning'], total = n_titles // batch_size + 1):
        train, valid = split_data(titles)
        #train = numpy.minimum(numpy.finfo(numpy.float32).max, train)
        #valid = numpy.minimum(numpy.finfo(numpy.float32).max, valid)
        valid_set.append(valid)

        #model.n_estimators += n_est_per_fit
        model.partial_fit(preproc.fit_transform(train[:,1:]), train[:,0], numpy.array([0, 1]))
    valid_set = numpy.vstack(valid_set)
    blackboard.LOG.info("Trained model AUC: {}".format(roc_auc_score(model.predict_proba(valid_set[:,1:])[:,1] > 0.5, valid_set[:,0])))
    del valid_set

    f.seek(0)
    header = next(f).split("\t")
    score_idx = header.index("score")
    f.seek(0)
    for titles, lines in tqdm.tqdm(load_data(f, batch_size), disable=log_level in ['fatal', 'error', 'warning'], total = n_titles // batch_size + 1, desc='Rescoring...'):
        data = numpy.vstack(list(titles.values()))
        #data = numpy.minimum(numpy.finfo(numpy.float32).max, data)
        scores = model.predict_proba(preproc.fit_transform(data[:,1:]))[:,0]
        idxs = numpy.argsort(scores)[::-1]
        data = data[idxs]
        scores = scores[idxs]
        for i, score in enumerate(scores):
            line = lines[i].strip().split("\t")
            yield line[:score_idx] + [score] + line[score_idx+1:]
