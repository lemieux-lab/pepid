# Small utility to output the pepid artifact filepaths implied by the given config.
# Useful for file management.

import sys
import os
import blackboard
import glob

filetypes = ['report', 'reportsvg', 'reportpkl', 'resultstsv', 'resultsdb', 'queriesdb', 'candsdb']

def check_input(argv):
    if len(argv) != 3 or argv[2] not in filetypes:
        return False
    return True

if __name__ == '__main__':
    if not check_input(sys.argv):
        print("Usage: {} config.cfg filetype\n\tfiletype: {}\n".format(sys.argv[0], " | ".join(filetypes)))
        sys.exit(-1)

    blackboard.config.read("data/default.cfg")
    blackboard.config.read(sys.argv[1])
    blackboard.setup_constants()

    filetype = sys.argv[2]

    # Note: the normal bash 'alternative' pattern ({a,b,c}) does not work with glob
    if filetype.startswith("report"):
        fname, fext = blackboard.config['data']['output'].rsplit(".", 1)
        pat = os.path.join(blackboard.config['report']['out'], "{}_{}_{}.{}")
        if filetype == "reportsvg":
            first = glob.glob(pat.format("plot", fname.rsplit("/", 1)[-1], "output", "svg"))
            second = glob.glob(pat.format("plot", fname.rsplit("/", 1)[-1], "rescored", "svg"))
            print("\n".join(first + second))
        elif filetype == "reportpkl":
            first = glob.glob(pat.format("report", fname.rsplit("/", 1)[-1], "output", "pkl"))
            second = glob.glob(pat.format("report", fname.rsplit("/", 1)[-1], "rescored", "pkl"))
            print("\n".join(first + second))
        elif filetype == "report":
            first = glob.glob(pat.format("report", fname.rsplit("/", 1)[-1], "output", "pkl"))
            second = glob.glob(pat.format("report", fname.rsplit("/", 1)[-1], "rescored", "pkl"))
            reports = first + second
            first = glob.glob(pat.format("plot", fname.rsplit("/", 1)[-1], "output", "svg"))
            second = glob.glob(pat.format("plot", fname.rsplit("/", 1)[-1], "rescored", "svg"))
            plots = first + second
            print("\n".join(reports + plots))
    elif filetype == "resultstsv":
        print(blackboard.config['data']['output'])
    elif filetype == "resultsdb":
        fname_pattern = list(filter(lambda x: len(x) > 0, blackboard.config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_*_pepidpart.sqlite"
        fname_path = "\n".join(glob.glob(os.path.join(blackboard.TMP_PATH, fname_pattern)))
        print(fname_path)
    elif filetype == "queriesdb":
        fname_pattern = list(filter(lambda x: len(x) > 0, blackboard.config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_q.sqlite"
        print(fname_pattern)
    elif filetype == "candsdb":
        fname_pattern = list(filter(lambda x: len(x) > 0, blackboard.config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_cands.sqlite"
        print(fname_pattern)
