import sys

if __package__ is None or __package__ == '':
    from pepid import pepid_utils
    from pepid import blackboard
else:
    from . import pepid_utils
    from . import blackboard

if __name__ == '__main__':
    blackboard.config.read(blackboard.here("data/default.cfg"))
    blackboard.config.read(sys.argv[1])

    blackboard.setup_constants()

    blackboard.LOCK = blackboard.acquire_lock()

    rescore_fn = pepid_utils.import_or(blackboard.config['rescoring']['function'].strip(), None)

    if rescore_fn is not None:
        f = open(blackboard.config['data']['output'], 'r')
        fname, ext = blackboard.config['data']['output'].rsplit('.', 1)
        outf = open(fname + blackboard.config['rescoring']['suffix'] + "." + ext, 'w')

        header = next(f)
        outf.write(header)
        f.close()

        import sys
        for data in rescore_fn(sys.argv[1]):
            outf.write("\t".join(list(map(str, data))) + "\n")

        outf.close()
    else:
        blackboard.LOG.warning("No rescoring function specified, not rescoring")
