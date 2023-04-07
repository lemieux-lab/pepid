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

FEATS_BLACKLIST = set(["seq", "modseq", "title", "desc", "decoy", "file"])

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

    tot_lines = count_lines(in_fname)
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
    scores = []

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
                    conn = sqlite3.connect("file:" + os.path.join(blackboard.TMP_PATH, prev_db + "_meta.sqlite") + "?cache=shared", detect_types=1, uri=True) 
                    cur = conn.cursor()

                cur.execute("SELECT rrow, data FROM meta WHERE rrow in ({}) ORDER BY score DESC;".format(",".join([p[rrow_idx] for p in payload])))
                metas = cur.fetchall()
                parsed_metas = []

                for i, (rrow, m) in enumerate(metas):
                    #meta = eval(m)
                    #meta = {k.strip()[1:-1] : float(v.strip()) for k, v in map(lambda x: x.strip().split(":"), m.strip()[1:-1].split(",")) if k.strip()[1:-1] in ALL_FEATS}
                    # The below is notably faster than eval() (about 2x here)
                    parsed_metas.append({k.strip()[1:-1] : numpy.minimum(numpy.finfo(numpy.float32).max, float(v.strip())) for k, v in map(lambda x: x.strip().split(":"), m.strip()[1:-1].split(",")) if k.strip()[1:-1] not in FEATS_BLACKLIST})
                    if feats is None:
                        feats = list(parsed_metas[-1].keys())
                        if 'score' in feats:
                            feats.append('deltLCn')
                    if 'score' in feats: # XXX: HACK for comet-like deltLCn feature should depend on config...
                        scores.append(parsed_metas[-1]['score'])

                blackboard.lock()
                pin = open(pin_name, 'a')

                for j, m in enumerate(parsed_metas):
                    if 'score' in feats:
                        m['deltLCn'] = (m['score'] - numpy.min(scores)) / (m['score'] if m['score'] != 0 else 1)

                    extraVals = "\t0.0\t0.0"
                    if 'expMass' in m and 'calcMass' in m:
                        extraVals = "\t{}\t{}".format(m['expMass'], m['calcMass'])

                    pin.write("{}\t{}\t{}{}".format(payload[j][title_idx], (1 - payload[j][desc_idx].startswith(decoy_prefix)) * 2 - 1, start+title_cnt, extraVals))

                    for k in feats:
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

    conn = sqlite3.connect("file:" + os.path.join(blackboard.TMP_PATH, first[header.index('file')] + "_meta.sqlite") + "?cache=shared", detect_types=1, uri=True) 
    cur = conn.cursor()
    cur.execute("SELECT data FROM meta LIMIT 1;")
    one = cur.fetchone()[0]
    feats = [k.strip()[1:-1] for k in map(lambda x: x.strip().split(":")[0], one.strip()[1:-1].split(","))]
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
