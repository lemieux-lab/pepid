import numpy
import sys
import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import tqdm
import math

from pyteomics import auxiliary

import blackboard

def tda_fdr(rescored=False):
    fname, fext = blackboard.config['data']['output'].rsplit(".", 1)
    full_fname = (fname + "." + fext) if not rescored else (fname + blackboard.config['rescoring']['suffix'] + "." + fext)
    decoy_prefix = blackboard.config['processing.db']['decoy prefix']
    topN = blackboard.config['report'].getint('max scores')

    data = []

    try:
        f = open(full_fname, 'r')
    except:
        import sys
        sys.stderr.write("FATAL: File specified in {} ({}) could not be read\n".format(sys.argv[1], full_fname))
        sys.exit(-1)
    for li, l in enumerate(f):
        if li > 0:
            try:
                title, desc, seq, modseq, score, db_prefix, db_line = l.split("\t", 6)
            except:
                import sys
                sys.stderr.write("During report: ERR: {}\n".format(l))
                sys.exit(-1)
            score = float(score)
            if not math.isinf(score):
                data.append((title, score, desc.startswith(decoy_prefix)))

    if len(data) == 0:
        import sys # despite top-level import, this is required... wtf???
        sys.stderr.write("FATAL: No entries in {}!\n".format(full_fname))
        sys.exit(-1)

    dtype = [('title', object), ('score', numpy.float64), ('decoy', bool)]
    ndata = numpy.array(data, dtype=dtype)
    ndata.sort(order=['score'])
    ndata = ndata[::-1]
    keys = numpy.unique([d[0] for d in data])
    grouped_data = {k: [] for k in keys}
    for d in ndata:
        if len(grouped_data[d['title']]) >= topN:
            continue
        grouped_data[d['title']].append(d)
    data = numpy.hstack([grouped_data[group] for group in grouped_data])

    fdr = (2 * data['decoy'].sum()) / float(len(data))

    # resort the collated subdata before further processing
    data.sort(order=['score'])
    data = data[::-1]

    fdr_index = numpy.arange(1, data.shape[0]+1)
    fdr_levels = numpy.cumsum(2 * data['decoy'].astype('float32')) / fdr_index
    sort_idx = numpy.argsort(fdr_levels)
    fdr_index = fdr_index[sort_idx]
    fdr_levels = fdr_levels[sort_idx]
    fmax = fdr_index[0]
    for i in range(len(fdr_index)):
        fmax = max(fmax, fdr_index[i])
        fdr_index[i] = fmax

    mask = numpy.logical_and(0.005 < fdr_levels, fdr_levels <= 0.25)
    fdr_index = fdr_index[mask]
    fdr_levels = fdr_levels[mask]

    if len(fdr_levels) == 0:
        blackboard.LOG.warning("Empty fdr levels in fdr report")
        fdr_levels = numpy.array([0])
        fdr_index = numpy.array([0])

    best_fdr_idx = -1
    for i, fv in enumerate(fdr_levels):
        if fv <= 0.01:
            best_fdr_idx = i
        else:
            break
    psm_at_1 = 0
    if best_fdr_idx >= 0:
       psm_at_1 = fdr_index[best_fdr_idx] 

    blackboard.LOG.info("Overall FDR: {}; FDR range: {}-{}; Peptide count over FDR range: {}-{}; PSM@1%: {}".format(fdr, fdr_levels.min(), fdr_levels.max(), fdr_index.min(), fdr_index.max(), psm_at_1))

    return len(grouped_data), fdr, numpy.array(list(zip(fdr_levels, fdr_index)))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("USAGE: {} config.cfg [output|rescored]\n".format(sys.argv[0]))
        sys.exit(1)

    blackboard.config.read("data/default.cfg")
    blackboard.config.read(sys.argv[1])

    blackboard.setup_constants()

    n_data, fdr, curve = tda_fdr(sys.argv[2].strip().lower() == 'rescored')
    fname, fext = blackboard.config['data']['output'].rsplit(".", 1)

    plt.title("Peptide Discovery vs FDR Threshold for {} ({})".format(sys.argv[1], sys.argv[2]))
    plt.ylabel("Discovered Peptides (Max: {})".format(n_data))
    plt.xlabel("FDR Threshold")
    plt.plot(curve[:,0], curve[:,1])
    plt.savefig(os.path.join(blackboard.config['report']['out'], "plot_{}_{}.svg".format(fname.rsplit("/", 1)[-1], sys.argv[2])))
