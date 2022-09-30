import blackboard
import sqlite3
import os
import tqdm
import numpy
import datetime
import subprocess
import re

#ALL_FEATS = set(["dM", "MIT", "MHT", "mScore",
#                        "peptideLength", "z1", "z2",
#                        "z4", "z7", "isoDM",
#                        "isoDMppm", "isoDmz",
#                        "12C", "mc0", "mc1", "mc2",
#                        'varmods',
#                        'varmodsCount', 'totInt',
#                       'intMatchedTot',
#                        'relIntMatchedTot', 'RMS',
#                        'RMSppm',
#                        'meanAbsFragDa', 'meanAbsFragPPM',
#                        'expMass', 'calcMass',
#                        'rawscore'])
FEATS_BLACKLIST = set(["seq", "modseq", "title", "desc", "decoy"])

def count_spectra(f):
    titles = {}
    _ = next(f)

    for l in f:
        title = l.split('\t', 1)[0]
        titles[title] = 1

    return len(titles)

def pout_to_tsv(psmout, psmout_decoys, scores_in):
    scores = {}
    for which in [psmout, psmout_decoys]:
        for il, l in enumerate(which):
            l = l.strip()
            if il == 0:
                header = l.split("\t")
                continue

            fields = l.split("\t")
            title = fields[header.index("PSMId")]
            score = fields[header.index("score")]
            seq = fields[header.index("peptide")][2:-2]

            if title not in scores:
                scores[title] = {}
            scores[title][seq] = score

        scores_in.seek(0)
        for il, l in enumerate(scores_in):
            if il == 0:
                header = l.strip().split("\t")
                score_idx = header.index("score")
                title_idx = header.index('title')
                seq_idx = header.index('modseq')
            else:
                line = l.strip().split("\t")
                title = line[title_idx]
                seq = line[seq_idx]
                if (title not in scores) or (seq not in scores[title]):
                    pass
                else:
                    yield line[:score_idx] + [scores[title][seq]] + line[score_idx+1:]

def generate_pin(fin, pin):
    n = count_spectra(fin)
    fin.seek(0)

    decoy_prefix = blackboard.config['processing.db']['decoy prefix']
    log_level = blackboard.config['logging']['level'].lower()

    header = next(fin).strip().split('\t')
    feats = None

    prev_db = None
    prev_title = None
    cnt = 0

    db_rows = []
    seqs = []
    descs = []
    titles = []
    #scores = []

    max_scores = blackboard.config['rescoring'].getint('max scores')

    def step(db_pre, idx):
        conn = sqlite3.connect("file:" + os.path.join(blackboard.TMP_PATH, db_pre + "_meta.sqlite") + "?cache=shared", detect_types=1, uri=True) 
        cur = conn.cursor()
        cur.execute("SELECT data FROM meta WHERE rrow in ({}) ORDER BY rrow;".format(",".join(list(map(str, db_rows)))))
        metas = cur.fetchall()
        reidx = numpy.argsort(db_rows)
        metas_sorted = [0] * len(metas)
        for imeta, meta in enumerate(metas):
            metas_sorted[reidx[imeta]] = meta[0]
        metas = metas_sorted

        scores = {}
        parsed_metas = []

        for i, m in enumerate(metas):
            #meta = eval(m)
            #meta = {k.strip()[1:-1] : float(v.strip()) for k, v in map(lambda x: x.strip().split(":"), m.strip()[1:-1].split(",")) if k.strip()[1:-1] in ALL_FEATS}
            # The below is notably faster than eval() (about 2x here)
            parsed_metas.append({k.strip()[1:-1] : numpy.minimum(numpy.finfo(numpy.float32).max, float(v.strip())) for k, v in map(lambda x: x.strip().split(":"), m.strip()[1:-1].split(",")) if k.strip()[1:-1] not in FEATS_BLACKLIST})
            if 'rawscore' in parsed_metas[-1]: # XXX: HACK for comet-like deltLCn feature should depend on config...
                if titles[i] not in scores:
                    scores[titles[i]] = []
                scores[titles[i]].append(parsed_metas[-1]['rawscore'])
                parsed_metas[-1]['deltLCn'] = 0

        i = -1
        seen = set()
        for j, m in enumerate(parsed_metas):
            if titles[j] not in seen:
                i += 1
            if 'rawscore' in m:
                m['deltLCn'] = (m['rawscore'] - numpy.min(scores[titles[j]])) / (m['rawscore'] if m['rawscore'] != 0 else 1)
            meta = m

            nonlocal feats

            extraFeats = ""
            extraVals = ""
            if 'expMass' in meta and 'calcMass' in meta:
                extraFeats = "ExpMass\tCalcMass\t" 
                extraVals = "\t{}\t{}".format(meta['expMass'], meta['calcMass'])

            if feats is None:
                feats = list(meta.keys())
                pin.write("PSMId\tLabel\tScanNr\t{}{}\tPeptide\tProteins\n".format(extraFeats, "\t".join(feats)))

            pin.write("{}\t{}\t{}{}".format(titles[j], (1 - descs[j].startswith(decoy_prefix)) * 2 - 1, idx + i + 1, extraVals))

            seen.add(titles[j])

            for k in feats:
                pin.write("\t{}".format(numpy.format_float_positional(meta[k], trim='0')))
            pin.write("\t{}\t{}\n".format("-." + seqs[j] + ".-", descs[j]))

    start_time = datetime.datetime.now()
    idx = 0
    title_cnt = 0
    dont_skip = True

    for il, l in enumerate(fin): # tqdm seems to be broken here for some reason... report performance caveman-style instead.
        sl = l.strip().split("\t")
        db_pre = sl[header.index('file')]
        title = sl[header.index('title')]
        desc = sl[header.index('desc')]
        #score = float(sl[header.index('score')])

        if title != prev_title:
            if prev_title is not None:
                cnt += 1
            prev_title = title
            title_cnt = 0
            dont_skip = True

        title_cnt += 1

        if db_pre != prev_db:
            if len(db_rows) != 0:
                step(prev_db, idx)
                idx += len(numpy.unique(titles))
                if log_level in ['debug', 'info']:
                    elapsed = datetime.datetime.now() - start_time
                    print("{}/{} ({}>{})".format(cnt, n, elapsed, (elapsed / (cnt / n)) - elapsed))
            prev_db = db_pre
            seqs = []
            descs = []
            db_rows = []
            titles = []
            #scores = []

        if dont_skip:
            db_rows.append(int(sl[header.index('rrow')]))
            descs.append(desc)
            seqs.append(sl[header.index('modseq')])
            titles.append(title)
            if title_cnt >= max_scores:
                dont_skip = False
        #scores.append(score)

    step(db_pre, idx)

def rescore(f):
    import blackboard
    import sys

    in_fname = blackboard.config['data']['output']
    fname, fext = in_fname.rsplit('.', 1)
    suffix = blackboard.config['rescoring']['suffix']
    pin_name = fname + suffix + "_pin.tsv"
    pout_name = fname + suffix + "_pout.xml"
    pepout_name = fname + suffix + "_pout_pep.tsv"
    psmout_name = fname + suffix + "_pout_psm.tsv"
    psmdout_name = fname + suffix + "_pout_decoys_psm.tsv"

    artifacts = [pin_name, pout_name, pepout_name, psmout_name, psmdout_name]

    blackboard.LOG.info("Generating percolator pin input...")
    if blackboard.config['percolator'].getboolean('generate pin'):
        pin_f = open(pin_name, "w")
        generate_pin(f, pin_f)
        pin_f.close()
    blackboard.LOG.info("Running percolator...")
    log_level = blackboard.config['logging']['level'].lower()
    extra_args = blackboard.config['percolator']['options'].split(" ")
    proc = subprocess.Popen([blackboard.config['percolator']['percolator'], pin_name, "-X", pout_name, "-Z", "-r", pepout_name, "-m", psmout_name, "-M", psmdout_name, "-v", "0" if log_level in ['fatal', 'error', 'warning'] else "2" if log_level in ['info'] else "2"] + (extra_args if len(extra_args) > 0 else []))
    while True:
        ret = proc.poll()
        if ret is not None:
            break

    blackboard.LOG.info("Percolator done; converting results to tsv...")
    f.seek(0)
    for l in pout_to_tsv(open(psmout_name, "r"), open(psmdout_name, 'r'), f):
        yield l

    if(blackboard.config['percolator'].getboolean('cleanup')):
        blackboard.LOG.info("Percolator: cleaning up")
        for a in artifacts:
            os.system("rm {}".format(a))
