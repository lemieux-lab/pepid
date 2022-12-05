import numpy
import sys
import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

plt.style.use('ggplot')
params = {
   'axes.labelsize': 12,
   'font.size': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   }
matplotlib.rcParams.update(params)

import tqdm
import math

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
        if li == 0:
            header = l.strip().split("\t")
        if li > 0:
            try:
                fields = l.strip().split("\t")
            except:
                import sys
                blackboard.LOG.error("During report: ERR: {}\n".format(l))
                sys.exit(-1)
            score = float(fields[header.index('score')])
            title = fields[header.index('title')]
            desc = fields[header.index('desc')]
            qrow = fields[header.index('qrow')]
            candrow = fields[header.index('candrow')]
            if not math.isinf(score):
                data.append((title, score, desc.startswith(decoy_prefix), qrow, candrow))

    if len(data) == 0:
        import sys # despite top-level import, this is required... wtf???
        blackboard.LOG.error("FATAL: No entries in {}!\n".format(full_fname))
        sys.exit(-1)

    dtype = [('title', object), ('score', numpy.float64), ('decoy', bool), ('qrow', numpy.int64), ('candrow', numpy.int64)]
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

    fdr = data['decoy'].sum() / numpy.logical_not(data['decoy']).sum()

    # resort the collated subdata before further processing
    data.sort(order=['score'])
    data = data[::-1]

    fdr_index = numpy.cumsum(numpy.logical_not(data['decoy']))
    fdr_levels = numpy.cumsum(data['decoy'].astype('float32')) / numpy.maximum(1, fdr_index)
    sort_idx = numpy.argsort(fdr_levels)
    fdr_index = fdr_index[sort_idx]
    fdr_levels = fdr_levels[sort_idx]
    fmax = fdr_index[0]
    for i in range(len(fdr_index)):
        fmax = max(fmax, fdr_index[i])
        fdr_index[i] = fmax

    if len(fdr_levels) == 0:
        blackboard.LOG.warning("Empty fdr levels in fdr report")
        fdr_levels = numpy.array([0])
        fdr_index = numpy.array([0])

    best_fdr_idx = -1
    fdr_limit = float(blackboard.config['report']['fdr threshold'])
    for i, fv in enumerate(fdr_levels):
        if fv <= fdr_limit:
            best_fdr_idx = i
        else:
            break
    psm_at_t = 0
    if best_fdr_idx >= 0:
       psm_at_t = fdr_index[best_fdr_idx] 

    blackboard.LOG.info("Overall FDR: {}; FDR range: {}-{}; Peptide count over FDR range: {}-{}; PSM@{}%: {}".format(fdr, fdr_levels.min(), fdr_levels.max(), fdr_index.min(), fdr_index.max(), int(fdr_limit * 100.), psm_at_t))

    return {
            'n_data': len(grouped_data),
            'fdr': fdr,
            'level': best_fdr_idx,
            'curve': numpy.array(list(zip(fdr_levels, fdr_index))),
            'decoy scores': data['score'][data['decoy']],
            'target scores': data['score'][numpy.logical_not(data['decoy'])],
            'spectra': data['qrow'],
            'peptides': data['candrow']
            }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        blackboard.LOG.error("USAGE: {} config.cfg [output|rescored]\n".format(sys.argv[0]))
        sys.exit(1)

    blackboard.config.read("data/default.cfg")
    blackboard.config.read(sys.argv[1])

    blackboard.setup_constants()

    stats = tda_fdr(sys.argv[2].strip().lower() == 'rescored')
    fname, fext = blackboard.config['data']['output'].rsplit(".", 1)

    all_scores = numpy.sort(numpy.hstack((stats['decoy scores'], stats['target scores'])))[::-1]
    score_limit = all_scores[stats['level']]
    fdr_limit = float(blackboard.config['report']['fdr threshold'])

    mosaic = """
AABBDD
AABBDD
EECC..
FFCC..
"""
    fig, axs = plt.subplot_mosaic(mosaic, figsize=(16,8), dpi=600)

    # FDR curve
    plt.title("{} ({})".format(sys.argv[1], sys.argv[2]))
    axs['A'].set_title("Peptide Discovery vs FDR Threshold")
    axs['A'].set_ylabel("Peptides Identified")
    axs['A'].set_xlabel("FDR Threshold")
    axs['A'].plot(stats['curve'][:,0], stats['curve'][:,1])

    # Score violins
    axs['B'].set_title("Score Violin Plots")
    axs['B'].set_ylabel("Score")
    axs['B'].set_xticks([1, 2])
    axs['B'].set_xticklabels(["Targets", "Decoys"])
    axs['B'].violinplot([stats['target scores'], stats['decoy scores']])

    # FDR/decoy rate over deciles
    axs['C'].set_title("Average FDR Over Score Deciles")
    axs['C'].set_ylabel("FDR")
    axs['C'].set_xlabel("Decile #")
    axs['C'].set_xticks(list(range(10)))
    axs['C'].set_xticklabels(list(map(str, range(1, 11))))

    dec_stats = [0 for _ in range(10)]
    inv_curve = stats['curve'][::-1,0]
    for i in range(10):
        start = int(i * 0.1 * len(stats['curve']))
        end = int((i+1) * 0.1 * len(stats['curve']))
        dec_stats[i] = inv_curve[start:end].mean()

    axs['C'].bar(list(range(10)), dec_stats)

    # Unique peptides, identified spectra
    axs['D'].set_title("Unique Peptides at {}% FDR".format(int(fdr_limit * 100)))
    axs['D'].set_xlabel("Count")
    axs['D'].set_yticks(list(range(2)))
    axs['D'].set_yticklabels(["Unique Peptide IDs", "All Peptide IDs"])
    axs['D'].barh(list(range(2)), [len(numpy.unique(stats['peptides'][stats['curve'][:,0] <= fdr_limit])), len(stats['peptides'][stats['curve'][:,0] <= fdr_limit])])

    # N decoys, N targets, N total at T% FDR
    n_decoys = (stats['decoy scores'] >= score_limit).sum()
    n_targets = (stats['target scores'] >= score_limit).sum()
    n_total = n_decoys + n_targets

    axs['E'].set_title("Hit Count by Type at {}% FDR".format(int(fdr_limit * 100)))
    axs['E'].set_yticks(list(range(2)))
    axs['E'].set_yticklabels(["Targets ({})".format(n_targets), "Decoys ({})".format(n_decoys)])
    axs['E'].tick_params(axis='x', rotation=90)
    axs['E'].barh([0, 1], [n_targets, n_decoys])

    # Identified spectra
    n_spectra_ids = len(numpy.unique(stats['spectra'][stats['curve'][:,0] <= fdr_limit]))
    n_spectra_no_ids = len(stats['curve']) - n_spectra_ids

    axs['F'].set_title("Identified Spectra at {}% FDR".format(int(fdr_limit * 100)))
    axs['F'].set_yticks(list(range(2)))
    axs['F'].set_yticklabels(["Spectra With IDs ({})".format(n_spectra_ids), "Spectra Without IDs ({})".format(n_spectra_no_ids)])
    axs['F'].tick_params(axis='x', rotation=90)
    axs['F'].barh([0, 1], [n_spectra_ids, n_spectra_no_ids])

    plt.tight_layout()
    plt.savefig(os.path.join(blackboard.config['report']['out'], "plot_{}_{}.svg".format(fname.rsplit("/", 1)[-1], sys.argv[2])))
