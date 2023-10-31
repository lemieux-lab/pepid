import numpy
import sys
import os
import tqdm
import math

if __package__ is None or __package__ == '':
    from pepid import blackboard
    from pepid import pepid_utils
else:
    from . import blackboard
    from . import pepid_utils

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

    staged = []
    prev_title = None
    for li, l in enumerate(f):
        if li == 0:
            header = l.strip().split("\t")
        else:
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
            charge = int(fields[header.index('query_charge')])
            mass = float(fields[header.index('query_mass')])
            seq = fields[header.index('seq')]
            modseq = fields[header.index('modseq')]
            if title != prev_title:
                prev_title = title
                if len(staged) > 0:
                    idxs = numpy.argsort([s[2] for s in staged])[::-1]
                    for n, idx in enumerate(idxs):
                        if n >= topN:
                            break
                        data.append(staged[idx])
                    del idxs
                    del staged
                    staged = []
            if not math.isinf(score):
                staged.append((title, modseq, score, desc.startswith(decoy_prefix), qrow, candrow, len(seq), charge, mass))

    idxs = numpy.argsort([s[2] for s in staged])[::-1]
    for n, idx in enumerate(idxs):
        if n >= topN:
            break
        data.append(staged[idx])
    del staged

    if len(data) == 0:
        import sys # despite top-level import, this is required... wtf???
        blackboard.LOG.error("FATAL: No entries in {}!\n".format(full_fname))
        sys.exit(-1)

    dtype = [('title', object), ('seq', object), ('score', numpy.float64), ('decoy', bool), ('qrow', numpy.int64), ('candrow', numpy.int64), ('lgt', numpy.int32), ('charge', numpy.int32), ('mass', numpy.float32)]
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

    fdrs = numpy.asarray(pepid_utils.calc_qval(data['score'], numpy.logical_not(data['decoy'])))

    if len(fdrs) == 0:
        blackboard.LOG.warning("Empty fdr levels in fdr report")
        fdrs = numpy.array([0])

    fdr_limit = float(blackboard.config['report']['fdr threshold'])
    aw = numpy.argwhere(fdrs <= fdr_limit)
    idx = min(len(data)-1, aw.reshape((-1,))[-1]) if len(aw) > 0 else -1

    blackboard.LOG.info("Overall FDR: {}; FDR range: {}-{}; PSM@{}%: {}".format(fdr, fdrs[0], fdrs[-1], int(fdr_limit * 100.), (data['score'] >= data['score'][idx]).sum() if idx >= 0 else 0))
    blackboard.LOG.info("Unique peps@{}%: {}".format(int(fdr_limit * 100.), len(numpy.unique(data[(data['score'] >= data['score'][idx])]['seq'])) if idx >= 0 else 0))

    ufdrs = numpy.unique(fdrs)
    levels = []
    for u in ufdrs:
        levels.append((fdrs <= u).sum())

    return {
            'n_data': len(grouped_data),
            'fdr': fdr,
            'level': idx,
            'curve': numpy.array(list(zip(ufdrs, levels))),
            'decoy scores': data['score'][data['decoy']],
            'target scores': data['score'][numpy.logical_not(data['decoy'])],
            'spectra': data['qrow'],
            'peptides': data['candrow'],
            'lgts': data['lgt'],
            'target lgts': data['lgt'][numpy.logical_not(data['decoy'])],
            'decoy lgts': data['lgt'][data['decoy']],
            'charges': data['charge'],
            'target charges': data['charge'][numpy.logical_not(data['decoy'])],
            'decoy charges': data['charge'][data['decoy']],
            'masses': data['mass'],
            'target masses': data['mass'][numpy.logical_not(data['decoy'])],
            'decoy masses': data['mass'][data['decoy']],
            'scores': data['score'],
            }

def plot_report(stats, fdr_limit, index=0, fig_axs=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    color_targets = "C{}".format(index*3)
    color_decoys = "C{}".format(index*3+1)
    color_all = "C{}".format(index*3+2)

    from matplotlib.lines import Line2D
    leg_lines = [Line2D([0], [0], color=color_all, label="Run {} (All)".format(index+1)), Line2D([0], [0], color=color_targets, label="Run {} (Targets)".format(index+1)), Line2D([0], [0], color=color_decoys, label="Run {} (Decoys)".format(index+1))]

    plt.style.use('ggplot')
    colors = plt.cm.tab20(numpy.linspace(0, 1, 18))
    params = {
       'font.size': 12,
       'axes.labelsize': 18,
       'legend.fontsize': 14,
       'xtick.labelsize': 16,
       'ytick.labelsize': 16,
       'text.usetex': False,
       'axes.prop_cycle': matplotlib.cycler(color=colors),
       }
    matplotlib.rcParams.update(params)

    if fig_axs is None:
        mosaic = """
AABBDD
AABBDD
EECCGG
FFCCGG
HHIIJJ
HHIIJJ
"""
        fig, axs = plt.subplot_mosaic(mosaic, figsize=(16,16), dpi=600)
    else:
        fig, axs = fig_axs

    all_scores = numpy.sort(numpy.hstack((stats['decoy scores'], stats['target scores'])))[::-1]
    score_limit = all_scores[stats['level']]

    # FDR curve
    axs['A'].set_title("Peptide Discovery vs FDR Threshold")
    axs['A'].set_ylabel("Peptides Identified")
    axs['A'].set_xlabel("FDR Threshold")
    axs['A'].plot(stats['curve'][:,0], stats['curve'][:,1], color=color_all)

    # Score violins
    axs['B'].set_title("Score Violin Plots")
    axs['B'].set_ylabel("Score")
    axs['B'].set_xticks([1, 2])
    axs['B'].set_xticklabels(["Targets", "Decoys"])
    vp = axs['B'].violinplot([stats['target scores'], stats['decoy scores']])
    vp['bodies'][0].set_color(color_targets)
    vp['bodies'][1].set_color(color_decoys)

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

    axs['C'].bar(list(range(10)), dec_stats, alpha=0.7, color=color_all)

    n_unique = len(numpy.unique(stats['peptides'][:stats['level']]))
    n_all = len(numpy.unique(stats['peptides']))

    # Unique peptides, identified spectra
    axs['D'].set_title("Unique Peptides ID'd at {}% FDR\nIDs: {} All: {}".format(int(fdr_limit * 100), n_unique, n_all))
    axs['D'].set_xlabel("Count")
    axs['D'].set_yticks(list(range(2)))
    axs['D'].set_yticklabels(["Unique Peptide IDs", "Unique Peptides"])
    axs['D'].barh(list(range(2)), [n_unique, n_all], alpha=0.7, color=color_all)
    axs['D'].tick_params(axis='x', rotation=90)

    # N decoys, N targets, N total at T% FDR
    n_decoys = (stats['decoy scores'] >= score_limit).sum()
    n_targets = (stats['target scores'] >= score_limit).sum()
    n_total = n_decoys + n_targets

    axs['E'].set_title("Hit Count by Type at {}% FDR\nTargets: {} Decoys: {}".format(int(fdr_limit * 100), n_targets, n_decoys))
    axs['E'].set_yticks(list(range(2)))
    axs['E'].set_yticklabels(["Targets", "Decoys"])
    axs['E'].tick_params(axis='x', rotation=90)
    axs['E'].barh([0, 1], [n_targets, n_decoys], alpha=0.7, color=[color_targets, color_decoys])

    # Identified spectra
    n_spectra_ids = len(numpy.unique(stats['spectra'][:stats['level']]))
    n_spectra_no_ids = len(numpy.unique(stats['spectra'])) - n_spectra_ids

    axs['F'].set_title("Identified Spectra at {}% FDR\nID: {} No ID: {}".format(int(fdr_limit * 100), n_spectra_ids, n_spectra_no_ids))
    axs['F'].set_yticks(list(range(2)))
    axs['F'].set_yticklabels(["Spectra With IDs", "Spectra Without IDs"])
    axs['F'].tick_params(axis='x', rotation=90)
    axs['F'].barh([0, 1], [n_spectra_ids, n_spectra_no_ids], alpha=0.7, color=color_all)

    # Score violins per decile
    min_score = min(stats['scores'])
    max_score = max(stats['scores'])
    axs['G'].set_title("Score Violin Plots -- Score Decile\n{:0.4f}-{:0.4f}".format(min_score, max_score))
    axs['G'].set_ylabel("Score")
    axs['G'].set_xlabel("Decile #")

    scores = stats['scores']
    deciles = numpy.percentile(scores, (numpy.arange(9) + 1)*10)
    all_scores = numpy.sort(stats['target scores'])
    dec_stats = [all_scores[numpy.logical_and((deciles[i-1] if i > 0 else 0) <= all_scores, all_scores < deciles[i])] for i in range(len(deciles))]
    dec_stats = [x if len(x) > 0 else numpy.array([0]) for x in dec_stats]

    axs['G'].set_xticks(list(range(1, len(dec_stats)+1)))
    axs['G'].set_xticklabels(list(map(str, range(1, len(dec_stats)+1))))

    target_violins = axs['G'].violinplot(dec_stats)

    for b in target_violins['bodies']:
        # get the center
        m = numpy.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], -numpy.inf, m)
        b.set_color(color_targets)

    all_scores = numpy.sort(stats['decoy scores'])
    dec_stats = [all_scores[numpy.logical_and((deciles[i-1] if i > 0 else 0) <= all_scores, all_scores < deciles[i])] for i in range(len(deciles))]
    dec_stats = [x if len(x) > 0 else numpy.array([0]) for x in dec_stats]
    dec_stats = [x if len(x) > 0 else numpy.array([0]) for x in dec_stats]

    decoy_violins = axs['G'].violinplot(dec_stats)

    for b in decoy_violins['bodies']:
        # get the center
        m = numpy.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further left than the center
        b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], m, numpy.inf)
        b.set_color(color_decoys)

    # Score violins per sequence length decile
    min_lgt = min(stats['lgts'])
    max_lgt = max(stats['lgts'])

    axs['H'].set_title("Score Violin Plots -- Sequence Length\n{}-{}".format(min_lgt, max_lgt))
    axs['H'].set_ylabel("Score")
    axs['H'].set_xlabel("Decile #")

    tgt_scores = stats['target scores']
    dec_scores = stats['decoy scores']
    scores = stats['scores']
    lengths = stats['lgts']
    tgt_lgts = stats['target lgts']
    dec_lgts = stats['decoy lgts']
    idxs = numpy.argsort(lengths)
    scores = scores[idxs]
    lengths = lengths[idxs]

    deciles = numpy.percentile(lengths, (numpy.arange(9) + 1)*10)
    dec_stats = [tgt_scores[numpy.logical_and((deciles[i-1] if i > 0 else 0) <= tgt_lgts, tgt_lgts < deciles[i])] for i in range(len(deciles))]
    dec_stats = [x if len(x) > 0 else numpy.array([0]) for x in dec_stats]

    axs['H'].set_xticks(list(range(1, len(dec_stats)+1)))
    axs['H'].set_xticklabels(list(map(str, range(1, len(dec_stats)+1))))

    target_violins = axs['H'].violinplot(dec_stats)

    for b in target_violins['bodies']:
        # get the center
        m = numpy.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], -numpy.inf, m)
        b.set_color(color_targets)

    dec_stats = [dec_scores[numpy.logical_and((deciles[i-1] if i > 0 else 0) <= dec_lgts, dec_lgts < deciles[i])] for i in range(len(deciles))]
    dec_stats = [x if len(x) > 0 else numpy.array([0]) for x in dec_stats]

    decoy_violins = axs['H'].violinplot(dec_stats)

    for b in decoy_violins['bodies']:
        # get the center
        m = numpy.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further left than the center
        b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], m, numpy.inf)
        b.set_color(color_decoys)

    # Score violins per charge decile
    min_charge = min(stats['charges'])
    max_charge = max(stats['charges'])

    axs['I'].set_title("Score Violin Plots -- Charge\n{}-{}".format(min_charge, max_charge))
    axs['I'].set_ylabel("Score")
    axs['I'].set_xlabel("Charge")

    tgt_charges = stats['target charges']
    dec_charges = stats['decoy charges']
    tgt_scores = stats['target scores']
    dec_scores = stats['decoy scores']
    tgt_dec_stats = []
    dec_dec_stats = []
    for z in range(min_charge, max_charge+1):
        tgt_s = tgt_scores[tgt_charges == z]
        if len(tgt_s) > 0:
            tgt_dec_stats.append(tgt_s)
        else:
            tgt_dec_stats.append(numpy.array([0]))

        dec_s = dec_scores[dec_charges == z]
        if len(dec_s) > 0:
            dec_dec_stats.append(dec_s)
        else:
            dec_dec_stats.append(numpy.array([0]))

    axs['I'].set_xticks(list(range(1, max_charge-min_charge+2)))
    axs['I'].set_xticklabels(list(map(str, range(min_charge, max_charge+1))))

    target_violins = axs['I'].violinplot(tgt_dec_stats)
    decoy_violins = axs['I'].violinplot(dec_dec_stats)

    for b in target_violins['bodies']:
        # get the center
        m = numpy.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], -numpy.inf, m)
        b.set_color(color_targets)

    for b in decoy_violins['bodies']:
        # get the center
        m = numpy.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further left than the center
        b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], m, numpy.inf)
        b.set_color(color_decoys)

    # Score violins per precursor mass decile
    min_mass = min(stats['masses'])
    max_mass = max(stats['masses'])

    axs['J'].set_title("Score Violin Plots -- Precursor Mass\n{:0.4f}-{:0.4f}".format(min_mass, max_mass))
    axs['J'].set_ylabel("Score")
    axs['J'].set_xlabel("Decile #")

    tgt_scores = stats['target scores']
    dec_scores = stats['decoy scores']
    scores = stats['scores']
    masses = stats['masses']
    tgt_masses = stats['target masses']
    dec_masses = stats['decoy masses']
    idxs = numpy.argsort(lengths)
    scores = scores[idxs]
    lengths = lengths[idxs]

    deciles = numpy.percentile(masses, (numpy.arange(9) + 1)*10)
    dec_stats = [tgt_scores[numpy.logical_and((deciles[i-1] if i > 0 else 0) <= tgt_masses, tgt_masses < deciles[i])] for i in range(len(deciles))]
    dec_stats = [x if len(x) > 0 else numpy.array([0]) for x in dec_stats]

    axs['J'].set_xticks(list(range(1, len(dec_stats)+1)))
    axs['J'].set_xticklabels(list(map(str, range(1, len(dec_stats)+1))))

    target_violins = axs['J'].violinplot(dec_stats)

    for b in target_violins['bodies']:
        # get the center
        m = numpy.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], -numpy.inf, m)
        b.set_color(color_targets)

    dec_stats = [dec_scores[numpy.logical_and((deciles[i-1] if i > 0 else 0) <= dec_masses, dec_masses < deciles[i])] for i in range(len(deciles))]
    dec_stats = [x if len(x) > 0 else numpy.array([0]) for x in dec_stats]

    decoy_violins = axs['J'].violinplot(dec_stats)

    for b in decoy_violins['bodies']:
        # get the center
        m = numpy.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further left than the center
        b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], m, numpy.inf)
        b.set_color(color_decoys)

    return fig, axs, leg_lines

if __name__ == "__main__":
    if len(sys.argv) != 3:
        blackboard.LOG.error("USAGE: {} config.cfg [output|rescored]\n".format(sys.argv[0]))
        sys.exit(-1)

    blackboard.config.read(blackboard.here("data/default.cfg"))
    blackboard.config.read(sys.argv[1])

    blackboard.setup_constants()

    stats = tda_fdr(sys.argv[2].strip().lower() == 'rescored')
    fname, fext = blackboard.config['data']['output'].rsplit(".", 1)

    fdr_limit = float(blackboard.config['report']['fdr threshold'])

    fig, axs, leg = plot_report(stats, fdr_limit)
    fig.legend(handles=leg, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.025))
    fig.tight_layout()

    import pickle
    if not os.path.exists(blackboard.config['report']['out']):
        os.makedirs(blackboard.config['report']['out'])
    report_pkl_path = os.path.join(blackboard.config['report']['out'], "report_{}_{}.pkl".format(fname.rsplit("/", 1)[-1], sys.argv[2]))
    pickle.dump(stats, open(report_pkl_path, "wb"))

    fig.savefig(os.path.join(blackboard.config['report']['out'], "plot_{}_{}.svg".format(fname.rsplit("/", 1)[-1], sys.argv[2])), bbox_inches='tight')
