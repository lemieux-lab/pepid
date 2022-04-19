import blackboard
import sqlite3
import os
import tqdm
import numpy
import datetime
import subprocess
import re

ALL_FEATS = set(["dM", "MIT", "MHT", "mScore",
                        "peptideLength", "z1", "z2",
                        "z4", "z7", "isoDM",
                        "isoDMppm", "isoDmz",
                        "12C", "mc0", "mc1", "mc2",
                        'varmods',
                        'varmodsCount', 'totInt',
                       'intMatchedTot',
                        'relIntMatchedTot', 'RMS',
                        'RMSppm',
                        'meanAbsFragDa', 'meanAbsFragPPM',
                        'expMass', 'calcMass',
                        'rawscore'])

def count_spectra(f):
    titles = {}
    _ = next(f)

    for l in f:
        title = l.split('\t', 1)[0]
        titles[title] = 1

    return len(titles)

def pout_to_tsv(pout, scores_in):
    process = False
    scores = {}
    for l in pout:
        l = l.strip()
        if not process:
            if l.startswith("<psms>"):
                process = True
            continue
        else:
            if l.startswith("</psms>"):
                break
            else:
                if l.startswith("<psm "):
                    title = re.sub(r"^.*p:psm_id=\"(.*)\" p:decoy=.*$", r"\1", l)
                elif l.startswith("<svm_score>"):
                    score = re.sub(r"^<svm_score>(.*)</svm_score>$", r"\1", l)
                elif l.startswith("<peptide_seq "):
                    seq = re.sub(r"^<peptide_seq n=\"\.\" c=\"\.\" seq=\"(.*)\"/>$", r"\1", l)
                    if title not in scores:
                        scores[title] = {}
                    scores[title][seq] = score
                else:
                    continue

    cnt = 0
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
                print("MISSING: {} {}".format(title, seq))
            else:
                cnt += 1
                print(cnt)
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
    db_rows = []
    cnt = 0

    seqs = []
    descs = []
    #scores = []

    def step(db_pre, idx):
        conn = sqlite3.connect("file:" + os.path.join(blackboard.TMP_PATH, db_pre + "_meta.sqlite") + "?cache=shared", detect_types=1, uri=True) 
        cur = conn.cursor()
        cur.execute("SELECT rrow, data FROM meta WHERE rrow in ({});".format(",".join(list(map(str, db_rows)))))
        metas = cur.fetchall()
        metas = numpy.array([m[1] for m in metas])[numpy.argsort([m[0] for m in metas])]

        for i, m in enumerate(metas):
            #meta = eval(m)
            meta = {k.strip()[1:-1] : float(v.strip()) for k, v in map(lambda x: x.strip().split(":"), m.strip()[1:-1].split(",")) if k.strip()[1:-1] in ALL_FEATS}
            nonlocal feats
            if feats is None:
                feats = list(meta.keys())
                pin.write("PSMId\tLabel\tScanNr\t{}\tPeptide\tProteins\n".format("\t".join(feats)))

            pin.write("{}\t{}\t{}".format(title, (1 - descs[i].startswith(decoy_prefix)) * 2 - 1, idx + i))
            for k in feats:
                pin.write("\t{}".format(numpy.format_float_positional(meta[k], trim='0')))
            #print(len(seqs), len(descs), len(db_rows), len(metas), i)
            pin.write("\t{}\t{}\n".format("." + seqs[i] + ".", descs[i]))

    start_time = datetime.datetime.now()
    idx = 0
    for il, l in enumerate(fin): # tqdm seems to be broken here for some reason... caveman mode engaged
        sl = l.strip().split("\t")
        db_pre = sl[header.index('file')]
        title = sl[0]
        desc = sl[1]
        #score = float(sl[header.index('score')])

        if title != prev_title:
            if prev_title is not None:
                cnt += 1
            prev_title = title

        if db_pre != prev_db:
            prev_db = db_pre
            if len(db_rows) != 0:
                #idxs = numpy.argsort(scores)[::-1]
                #seqs = numpy.array(seqs)[idxs]
                #db_rows = numpy.array(db_rows)[idxs]
                #descs = numpy.array(descs)[idxs]
                step(db_pre, idx)
                idx += len(db_rows)
                if log_level in ['debug', 'info']:
                    elapsed = datetime.datetime.now() - start_time
                    print("{}/{} ({}>{})".format(cnt, n, elapsed, (elapsed / (cnt / n)) - elapsed))
            seqs = []
            descs = []
            db_rows = []
            #scores = []

        db_rows.append(int(sl[header.index('rrow')]))
        descs.append(desc)
        seqs.append(sl[3])
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
    #proc = subprocess.Popen([blackboard.config['percolator']['percolator'], pin_name, "-X", pout_name, "-Z", "-r", pepout_name, "-m", psmout_name, "--trainFDR", "0.1", "--testFDR", "0.1", "-v", "0" if log_level in ['fatal', 'error', 'warning'] else "2" if log_level in ['info'] else "2"])
    proc = subprocess.Popen([blackboard.config['percolator']['percolator'], pin_name, "-X", pout_name, "-Z", "-r", pepout_name, "-m", psmout_name, "-M", psmdout_name, "--trainFDR", "0.2", "--testFDR", "0.2", "-v", "0" if log_level in ['fatal', 'error', 'warning'] else "2" if log_level in ['info'] else "2"])
    while True:
        ret = proc.poll()
        if ret is not None:
            break

    blackboard.LOG.info("Percolator done; converting results to tsv...")
    f.seek(0)
    for l in pout_to_tsv(open(pout_name, "r"), f):
        yield l

    if(blackboard.config['percolator'].getboolean('cleanup')):
        blackboard.LOG.info("Percolator: cleaning up")
        for a in artifacts:
            os.system("rm {}".format(a))
