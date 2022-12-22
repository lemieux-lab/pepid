import sqlite3
import os
import numpy
import subprocess
import re
import math

import blackboard
import pepid_mp

import struct
import math

FEATS_BLACKLIST = set(["seq", "modseq", "title", "desc", "decoy", "file"])

def count_spectra(f):
    titles = set()

    for il, l in enumerate(f):
        if il == 0:
            header = l.strip().split("\t")
        else:
            title = l.strip().split('\t')[header.index('qrow')]
            titles.add(title)

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

def generate_pin(start, end):
    in_fname = blackboard.config['data']['output']
    fname, fext = in_fname.rsplit('.', 1)
    suffix = blackboard.config['rescoring']['suffix']
    pin_name = fname + suffix + "_pin.tsv"

    fin = open(in_fname, 'r')

    decoy_prefix = blackboard.config['processing.db']['decoy prefix']
    log_level = blackboard.config['logging']['level'].lower()

    header = next(fin).strip().split('\t')

    prev_db = None
    prev_title = None
    cnt = 0

    max_scores = blackboard.config['rescoring'].getint('max scores')

    dtype = [('rrow', numpy.int32), ('desc', numpy.object), ('seq', numpy.object), ('title', numpy.object), ('db', numpy.object)]
    keys = [d[0] for d in dtype]

    feats = None

    def step(payload, idx):
        payload = [tuple([p[k] for k in keys]) for p in payload]
        payload = numpy.array(payload, dtype=dtype)
        payload = numpy.sort(payload, order='db')

        prev_file = None
        rrows = []
        titles = []
        descs = []
        seqs = []

        scores = {}
        parsed_metas = []

        for ip, p in enumerate(payload):
            if p['db'] != prev_file or ip == len(payload)-1:
                if prev_file is not None:
                    cur.execute("SELECT rrow, data FROM meta WHERE rrow in ({}) ORDER BY rrow;".format(",".join(list(map(str, rrows)))))
                    metas = cur.fetchall()
                    parsed_metas = []

                    for i, (rrow, m) in enumerate(metas):
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

                    blackboard.lock()
                    pin = open(pin_name, 'a')

                    for j, m in enumerate(parsed_metas):
                        if titles[j] not in seen:
                            i += 1
                            seen.add(titles[j])
                        if 'rawscore' in m:
                            m['deltLCn'] = (m['rawscore'] - numpy.min(scores[titles[j]])) / (m['rawscore'] if m['rawscore'] != 0 else 1)

                        extraVals = "\t0.0\t0.0"
                        if 'expMass' in m and 'calcMass' in m:
                            extraVals = "\t{}\t{}".format(m['expMass'], m['calcMass'])

                        nonlocal feats
                        if feats is None:
                            feats = list(m.keys())

                        pin.write("{}\t{}\t{}{}".format(titles[j], (1 - descs[j].startswith(decoy_prefix)) * 2 - 1, idx + i, extraVals))

                        for k in feats:
                            pin.write("\t{}".format(numpy.format_float_positional(m[k], trim='0')))
                        pin.write("\t{}\t{}\n".format("-." + seqs[j] + ".-", descs[j]))

                    pin.close()
                    blackboard.unlock()

                    cur.close()
                    conn.close()
                    rrows = []
                    titles = []
                    descs = []
                    seqs = []
                prev_file = p['db']
                conn = sqlite3.connect("file:" + os.path.join(blackboard.TMP_PATH, p['db'] + "_meta.sqlite") + "?cache=shared", detect_types=1, uri=True) 
                cur = conn.cursor()

            titles.append(p['title'])
            rrows.append(p['rrow'])
            seqs.append(p['seq'])
            descs.append(p['desc'])

        return idx + i + 1

    title_cnt = 0
    dont_skip = True
    prev_cnt = 0
    cnt = 0

    payload = []

    idx = 0

    for il, l in enumerate(fin):
        if il < start:
            continue
        if il >= end:
            break
        if len(payload) > 0 and len(payload) % 10000 == 0:
            idx = step(payload, idx)
            payload = []
        sl = l.strip().split("\t")
        db_pre = sl[header.index('file')]
        title = sl[header.index('title')]
        desc = sl[header.index('desc')]

        if title != prev_title:
            if prev_title is not None:
                cnt += 1
            prev_title = title
            title_cnt = 0
            dont_skip = True

        title_cnt += 1

        if dont_skip:
            payload.append({'rrow': int(sl[header.index('rrow')]), 'desc': desc, 'seq': sl[header.index('modseq')], 'title': title, 'db': db_pre})
            if title_cnt > max_scores:
                dont_skip = False

    if len(payload) > 0:
        idx = step(payload, idx)

    fin.close()

def count_lines(f):
    ff = open(f, 'r')
    for il, l in enumerate(ff): pass
    ff.close()
    return il

def rescore(cfg_file):
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
        n_total = count_lines(in_fname)
        n_batches = math.ceil(n_total / batch_size)
        spec = [("pin_node.py", nworkers, n_batches,
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
    for l in pout_to_tsv(open(psmout_name, "r"), open(psmdout_name, 'r'), open(in_fname, 'r')):
        yield l

    if(blackboard.config['rescoring.percolator'].getboolean('cleanup')):
        blackboard.LOG.info("Percolator: cleaning up")
        for a in artifacts:
            os.system("rm {}".format(a))
