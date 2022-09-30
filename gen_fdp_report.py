import numpy
import sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import blackboard

def tda_fdr():
    f = blackboard.config['data']['output']
    decoy_prefix = blackboard.config['decoys']['decoy prefix']

    data = []

    f = open(f, 'r')
    for li, l in enumerate(f):
        if li > 0:
            title, desc, seq, modseq, cmass, mass, charge, score = l.split("\t")
            data.append([title, float(score), desc.startswith(decoy_prefix)])

    data = numpy.array(data, dtype=[('title', numpy.unicode_, 1), ('score', numpy.float32, (1,)), ('decoy', numpy.boolean, (1,))])

    return data

if __name__ == __main__:
    if len(sys.argv) != 4:
        print("USAGE: {} config.cfg topN out\n\ttopN: Use top N hits per query for report\n\tout: output directory".format(sys.argv[0]))
        sys.exit(1)

    blackboard.config.read("data/default.cfg")
    blackboard.config.read(sys.argv[1])
    results = tda_fdr(int(sys.argv[3]))

    plt.plot(results['score'])
    plt.savefig(os.path.join(sys.argv[4], "plot.svg"))


