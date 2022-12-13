import gen_fdr_report
import pickle
import sys
import os
import blackboard

blackboard.config.read("data/default.cfg")
blackboard.config.read(sys.argv[1])
blackboard.setup_constants()
fdr_limit = float(blackboard.config['report']['fdr threshold'])

f1 = pickle.load(open(sys.argv[2], 'rb'))
f2 = pickle.load(open(sys.argv[3], 'rb'))

fig, axs, leg1 = gen_fdr_report.plot_report(f1, fdr_limit, index=0)
fig, axs, leg2 = gen_fdr_report.plot_report(f2, fdr_limit, index=1, fig_axs=(fig, axs))
fig.legend(handles=leg1+leg2, loc='lower center', ncols=4, bbox_to_anchor=(0.5, -0.025))
fig.tight_layout()

base = blackboard.config['data']['output'].rsplit(".", 1)[0]
fname1 = sys.argv[2].rsplit("/", 1)[-1].rsplit(".", 1)[0]
fname2 = sys.argv[3].rsplit("/", 1)[-1].rsplit(".", 1)[0]
fig.savefig(os.path.join(blackboard.config['report']['out'], "plot_compare_{}_{}_{}.svg".format(base, fname1, fname2)), bbox_inches='tight')
