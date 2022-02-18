import sys
import pepid_utils
import blackboard

if __name__ == '__main__':
    blackboard.config.read("data/default.cfg")
    blackboard.config.read(sys.argv[1])

    rescore_fn = pepid_utils.import_or(blackboard.config['pipeline']['rescorer'], None)

    if rescore_fn is not None:
        rescore_fn()
