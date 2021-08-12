import sys
import blackboard

if __name__ == '__main__':
    rescore_fn = None

    blackboard.config.read("data/default.cfg")
    blackboard.config.read(sys.argv[1])

    try:
        mod, fn = blackboard.config['pipeline']['rescorer'].rsplit('.', 1)
        user_fn = getattr(__import__(mod, fromlist=[fn]), fn)
        rescore_fn = user_fn
    except:
        sys.stderr.write("[rescore]: user rescore function not found\n")
        sys.exit(-1)

    rescore_fn()
