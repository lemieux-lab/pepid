import blackboard
import sys

def rescore():
    blackboard.config.read("data/default.cfg")
    blackboard.config.read(sys.argv[1])

    in_f = blackboard.config['data']['output']
    fname, fext = in_f.rsplit('.', 1)
    f = open(in_f, 'r')
    suffix = blackboard.config['pipeline']['rescore suffix']
    out = open(fname + suffix + "." + fext, "w")

    for li, l in enumerate(f):
        if li == 0:
            out.write(l.strip() + "; final score\n")
        else:
            line = l.strip()
            _, params = l.rsplit(";", 1)
            params = eval(params)
            out.write(line + ";" + str(params['mScore']) + "\n")
