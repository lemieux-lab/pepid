import sys
import pepid_utils
import blackboard

if __name__ == '__main__':
    blackboard.config.read("data/default.cfg")
    blackboard.config.read(sys.argv[1])

    blackboard.setup_constants()

    rescore_fn = pepid_utils.import_or(blackboard.config['rescoring']['function'], None)

    if rescore_fn is not None:
        f = open(blackboard.config['data']['output'], 'r')
        fname, ext = blackboard.config['data']['output'].rsplit('.', 1)
        outf = open(fname + blackboard.config['rescoring']['suffix'] + "." + ext, 'w')

        header = next(f)
        outf.write(header)
        f.seek(0)

        for data in rescore_fn(sys.argv[1]):
            outf.write("\t".join(list(map(str, data))) + "\n")

        f.close()
        outf.close()
    else:
        blackboard.LOG.warning("No rescoring function specified, not rescoring")
