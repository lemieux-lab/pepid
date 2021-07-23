import blackboard
import os
import time
import numpy

def write_output():
    """
    Exports search results to csv format as per the config settings.
    """
    import glob
    files = glob.glob(os.path.join(blackboard.config['data']['tmpdir'], "results*.npy"))
    for f in files:
        results = numpy.memmap(f, dtype=blackboard.RES_DTYPE, shape=blackboard.config['performance'].getint('batch size'), mode='r')

        max_cands = blackboard.config['search'].getint('max retained candidates')

        out_fname = blackboard.config['data']['output']
        f = open(out_fname, 'w')

        header = list(map(lambda x: x[0], blackboard.RES_DTYPE))

        f.write(";".join(header) + "\n")

        for res in results:
            if res[header.index('score')] > 0:
                for i, r in enumerate(res):
                    if i != header.index('modseq'):
                        f.write(str(r).replace(";", ","))
                    else:
                        f.write("".join([s if m == 0 else s + "[{}]".format(m) for s,m in zip(res[header.index('seq')], r)]))
                    if i < len(res)-1:
                        f.write(";")
                    else:
                        f.write("\n")
        f.close()

    #results.close()
