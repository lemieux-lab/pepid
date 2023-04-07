import sys
import os
import glob
import math
import struct

if __package__ is None or __package__ == '':
    import blackboard
    import pepid_mp
else:
    from . import blackboard
    from . import pepid_mp

if __name__ == "__main__":
    cfg_file = sys.argv[1]

    blackboard.config.read(blackboard.here("data/default.cfg"))
    blackboard.config.read(cfg_file)
    blackboard.setup_constants()

    fname_pattern = list(filter(lambda x: len(x) > 0, blackboard.config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_*_pepidpart.sqlite"
    fname_path = os.path.join(blackboard.TMP_PATH, fname_pattern)

    batch_size = blackboard.config['postsearch'].getint('batch size')
    log_level = blackboard.config['logging']['level'].lower()

    all_files = glob.glob(fname_path)

    n_total = len(all_files)

    n_batches = math.ceil(n_total / batch_size)
    nworkers = blackboard.config['postsearch'].getint('workers')
    spec = [(blackboard.here("post_node.py"), nworkers, n_batches,
                    [struct.pack("!cI{}sc".format(len(blackboard.TMP_PATH)), bytes([0x00]), len(blackboard.TMP_PATH), blackboard.TMP_PATH.encode("utf-8"), "$".encode("utf-8")) for _ in range(nworkers)],
                    [struct.pack("!cQQc", bytes([0x01]), b * batch_size, min((b+1) * batch_size, n_total), "$".encode("utf-8")) for b in range(n_batches)],
                    [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(nworkers)])]


    pepid_mp.handle_nodes("Postsearch", spec, cfg_file=cfg_file, tqdm_silence=log_level in ['fatal', 'error', 'warning'])
