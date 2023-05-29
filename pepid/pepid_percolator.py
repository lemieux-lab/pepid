import sqlite3
import os
import numpy
import subprocess
import re
import math
import msgpack

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

    use_extra = blackboard.config['rescoring.percolator'].getboolean('use extra')

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
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT data, extra FROM meta LIMIT 1;")
    res = cur.fetchone()
    first = msgpack.loads(res['data'])
    if use_extra:
        second = msgpack.loads(res['extra'])
        if second is not None:
            first = {**first, **second}
    feats = sorted(list(first.keys()))
    if 'score' in feats:
        feats.append('deltLCn')
    feats = [x for x in feats if x not in blackboard.FEATS_BLACKLIST]

    cur.close()
    conn.close()

    log_level = blackboard.config['logging']['level'].lower()
    if blackboard.config['misc.tsv_to_pin'].getboolean('enabled'):
        fpin = open(pin_name, 'w')
        pin_template = blackboard.pin_template()
        pin_template[pin_template.index("FEATURES")] = "\t".join(feats)
        fpin.write("\t".join(pin_template) + "\n")
        fpin.close()

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
