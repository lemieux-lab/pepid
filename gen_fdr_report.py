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
    f = (fname + "." + fext) if not rescored else (fname + blackboard.config['pipeline']['rescore suffix'] + "." + fext)
    decoy_prefix = blackboard.config['decoys']['decoy prefix']
    topN = blackboard.config['report'].getint('retain')

    data = []

    f = open(f, 'r')
    for li, l in enumerate(f):
        if li > 0:
            title, desc, seq, modseq, cmass, mass, rt, charge, score, score_parts = l.split(";", 9)
            if not rescored:
                score = float(score)
            else:
                score = float(score_parts.split(";")[1])
            if not math.isinf(score):
                data.append((title, score, desc.startswith(decoy_prefix)))

    dtype = [('title', numpy.unicode_, 1024), ('score', numpy.float64, 1), ('decoy', numpy.bool, 1)]
    data = numpy.array(data, dtype=dtype)
    data.sort(order=['score'])
    data = data[::-1]
    keys = numpy.unique(data['title'])
    grouped_data = {k: [] for k in keys}
    for d in data:
        if len(grouped_data[d['title']]) >= topN:
            continue
        grouped_data[d['title']].append(d)
    data = numpy.hstack([grouped_data[group] for group in grouped_data])

    fdr = (2 * data['decoy'].sum()) / float(len(data))

    data.sort(order=['score'])
    data = data[::-1]
    fdr_index = numpy.arange(1, data.shape[0]+1)
    fdr_levels = numpy.cumsum(2 * data['decoy'].astype('float32')) / fdr_index
    sort_idx = numpy.argsort(fdr_levels)
    fdr_index = fdr_index[sort_idx]
    fdr_levels = fdr_levels[sort_idx]

    mask = numpy.logical_and(0.005 < fdr_levels, fdr_levels <= 0.25)
    fdr_index = fdr_index[mask] - 1
    fdr_levels = fdr_levels[mask]

    if len(fdr_levels) == 0:
        print("[WARN] Empty fdr levels in fdr report")
        fdr_levels = numpy.array([0])
        fdr_index = numpy.array([0])

    print(fdr, fdr_levels.min(), fdr_levels.max(), fdr_index.min(), fdr_index.max())

    return len(data), fdr, numpy.array(list(zip(fdr_index, fdr_levels)))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: {} config.cfg [output|rescored]".format(sys.argv[0]))
        sys.exit(1)

    blackboard.config.read("data/default.cfg")
    blackboard.config.read(sys.argv[1])
    n_data, fdr, curve = tda_fdr()

    plt.title("Peptide Discovery vs FDR Threshold for {} ({})".format(sys.argv[1], sys.argv[2]))
    plt.ylabel("Discovered Peptides (Max: {})".format(n_data))
    plt.xlabel("FDR Threshold")
    plt.plot(curve[:,0], curve[:,1])
    plt.savefig(os.path.join(blackboard.config['report']['out'], "plot_{}.svg".format(sys.argv[2])))
