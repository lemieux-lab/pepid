import sqlite3
import os
import numpy
import subprocess
import re
import math

if __package__ is None or __package__ == '':
    import blackboard
    import pepid_mp
    import queries
else:
    from . import blackboard
    from . import pepid_mp
    from . import queries

import struct
import math

FEATS_BLACKLIST = set([])

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
                    continue
                else:
                    yield line[:score_idx] + [scores[title][seq]] + line[score_idx+1:]

def generate_pin(start, end):
    in_fname = blackboard.config['data']['output']
    fname, fext = in_fname.rsplit('.', 1)
    suffix = blackboard.config['rescoring']['suffix']
    pin_name = fname + suffix + "_pin.tsv"

    tot_lines = count_lines(in_fname) - 1 # we will be skipping the first line
    fin = open(in_fname, 'r')

    decoy_prefix = blackboard.config['processing.db']['decoy prefix']
    log_level = blackboard.config['logging']['level'].lower()

    header = next(fin).strip().split('\t')
    score_idx = header.index('score')
    file_idx = header.index('file')
    rrow_idx = header.index('rrow')
    qrow_idx = header.index('qrow')
    seq_idx = header.index('modseq')
    title_idx = header.index('title')
    desc_idx = header.index('desc')

    prev_db = None
    prev_title = None
    cnt = 0

    title_cnt = -1 # +1 happens before lines are added, so have to -1 start at 0 (-1+1)

    max_scores = blackboard.config['rescoring'].getint('max scores')

    prev_q = None
    prev_db = None
    payload = []

    conn = None
    cur = None

    feats = None
    feats_raw = None
    scores = []

    score_feat_idx = None
    for il, l in enumerate(fin):
        line = l.strip().split("\t")

        if prev_q != line[qrow_idx] or (il == tot_lines-1):
            if il == tot_lines-1:
                payload.append(line)
            if len(payload) > 0:
                idxs = numpy.argsort([float(p[score_idx]) for p in payload])[::-1][:max_scores]
                payload = [payload[idx] for idx in idxs]
                if payload[0][file_idx] != prev_db:
                    prev_db = payload[0][file_idx]
                    if conn is not None:
                        cur.close()
                        conn.close()
                    conn = sqlite3.connect("file:" + os.path.join(blackboard.TMP_PATH, prev_db + ".sqlite") + "?cache=shared", detect_types=1, uri=True) 
                    conn.row_factory = sqlite3.Row
                    cur = conn.cursor()
                    if feats is None:
                        blackboard.execute(cur, "SELECT * FROM results LIMIT 1;")
                        keys = dict(cur.fetchone()).keys()
                        feats_raw = sorted([k for k in keys if k.startswith("META_")])
                        feats = [k[len("META_"):] for k in feats_raw]
                        if 'META_score' in feats_raw:
                            score_feat_idx = feats_raw.index('META_score')
                        elif 'score' in feats_raw:
                            feats.append('score')
                            score_feat_idx = feats_raw.index('score')
                        if 'score' in feats:
                            feats.append('deltLCn')

                blackboard.execute(cur, "SELECT {} FROM results WHERE rrow in ({}) ORDER BY score DESC;".format(",".join(feats_raw), ",".join([p[rrow_idx] for p in payload])))
                metas = [dict(m) for m in cur.fetchall()]

                blackboard.lock()
                pin = open(pin_name, 'a')

                if 'score' in feats:
                    for m in metas:
                        scores.append(m[feats_raw[score_feat_idx]])

                for j, m in enumerate(metas):
                    if 'score' in feats:
                        m['deltLCn'] = float((m[feats_raw[score_feat_idx]] - numpy.min(scores)) / (m[feats_raw[score_feat_idx]] if m[feats_raw[score_feat_idx]] != 0 else 1))

                    extraVals = "\t0.0\t0.0"
                    if 'META_expMass' in m and 'META_calcMass' in m:
                        extraVals = "\t{}\t{}".format(m['META_expMass'], m['META_calcMass'])

                    pin.write("{}\t{}\t{}{}".format(payload[j][title_idx], (1 - payload[j][desc_idx].startswith(decoy_prefix)) * 2 - 1, start+title_cnt, extraVals))

                    for k in feats_raw + (['deltLCn'] if 'score' in feats else []):
                        if m[k] is not None and type(m[k]) != str:
                            assert type(m[k]) in [int, float], "WTF {} {} {}".format(k, m[k], m)
                            pin.write("\t{}".format(numpy.format_float_positional(m[k], trim='0', precision=12))) # percolator breaks if too many digits are provided
                    pin.write("\t{}\t{}\n".format("-." + payload[j][seq_idx] + ".-", payload[j][desc_idx]))

                pin.close()
                blackboard.unlock()

            prev_q = line[header.index('qrow')]
            payload = []
            scores = []
            title_cnt += 1

            if title_cnt >= end:
                break

        if title_cnt >= start:
            payload.append(line)

    fin.close()

def count_lines(f):
    ff = open(f, 'r')
    for il, l in enumerate(ff): pass
    ff.close()
    return il

def rescore(cfg_file):
    if __package__ is None or __package__ == '':
        import blackboard
    else:
        from . import blackboard
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

    # Dirty little hack to get the header right before we start multiprocessing
    fin = open(in_fname, 'r')
    header = next(fin).strip().split('\t')
    first = next(fin).strip().split('\t')
    fin.close()

    conn = sqlite3.connect("file:" + os.path.join(blackboard.TMP_PATH, first[header.index('file')] + ".sqlite") + "?cache=shared", detect_types=1, uri=True) 
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM results LIMIT 1;")
    keyvals = [(k, v) for k, v in dict(cur.fetchone()).items()]
    keys = [kv[0] for kv in keyvals]
    feats = sorted([keyvals[k][0] for k in range(len(keyvals)) if keyvals[k][0].startswith('META_') and type(keyvals[k][1]) in set([int, float])])
    feats = [f[len('META_'):] for f in feats]
    for extra in ['score']:
        if extra in keys and extra not in feats:
            feats.append(extra)
    if 'score' in feats:
        feats.append('deltLCn')
    feats = [x for x in feats if x not in FEATS_BLACKLIST]

    fpin = open(pin_name, 'w')
    fpin.write("PSMId\tLabel\tScanNr\texpMass\tcalcMass\t{}\tPeptide\tProteins\n".format("\t".join(feats)))
    fpin.close()

    cur.close()
    conn.close()

    log_level = blackboard.config['logging']['level'].lower()
    if blackboard.config['rescoring.percolator'].getboolean('generate pin'):
        nworkers = blackboard.config['rescoring.percolator'].getint('pin workers')
        batch_size = blackboard.config['rescoring.percolator'].getint('pin batch size')
        n_total = queries.count_queries()
        n_batches = math.ceil(n_total / batch_size)
        spec = [(blackboard.here("pin_node.py"), nworkers, n_batches,
                        [struct.pack("!cI{}sc".format(len(blackboard.TMP_PATH)), bytes([0x00]), len(blackboard.TMP_PATH), blackboard.TMP_PATH.encode("utf-8"), "$".encode("utf-8")) for _ in range(nworkers)],
                        [struct.pack("!cQQc", bytes([0x01]), b * batch_size, min((b+1) * batch_size, n_total), "$".encode("utf-8")) for b in range(n_batches)],
                        [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(nworkers)])]

        pepid_mp.handle_nodes("PIN Generation", spec, cfg_file=cfg_file, tqdm_silence=log_level in ['fatal', 'error', 'warning'])

    blackboard.LOG.info("Running percolator...")
    extra_args = blackboard.config['rescoring.percolator']['options'].split(" ")
    proc = subprocess.Popen([blackboard.config['rescoring.percolator']['percolator'], pin_name, "-X", pout_name, "-Z", "-r", pepout_name, "-m", psmout_name, "-M", psmdout_name, "-v", "0" if log_level in ['fatal', 'error', 'warning'] else "2" if log_level in ['info'] else "2"] + (extra_args if len(extra_args) > 0 else []))
    while True:
        ret = proc.poll()
        if ret is not None:
            break

    blackboard.LOG.info("Percolator done; converting results to tsv...")
    f1, f2, f3 = open(psmout_name, "r"), open(psmdout_name, 'r'), open(in_fname, 'r')
    for l in pout_to_tsv(f1, f2, f3):
        yield l
    f1.close()
    f2.close()
    f3.close()

    if(blackboard.config['rescoring.percolator'].getboolean('cleanup')):
        blackboard.LOG.info("Percolator: cleaning up")
        for a in artifacts:
            os.system("rm {}".format(a))
