import time
import sys
import tqdm
import numpy
import pickle
import glob

import blackboard

import logging
import subprocess
import os
import socket
import select
import struct
import math
import uuid
import tempfile
import queue
import copy

# exit codes:
# 0: success
# 1: args incorrect
# 2: invalid config parameter (fatal)

def handle_response(resp):
    ret = []
    template = {"resp": False, "code": None, "data": None, "error": True}
    if not resp:
        return ret
    else:
        while True:
            this_ret = copy.copy(template) 
            this_ret["code"] = resp[0]
            this_ret["data"] = resp[1:3]
            skip_lgt = 1 + 1 + 1
            if resp[0] == 0xfd:
                msg_lgt = struct.unpack("!I", resp[1:5])[0]
                log.error("Node: {}".format(resp[1+4:1+4+msg_lgt].decode('utf-8')))
                this_ret['error'] = True
                skip_lgt = 1+4+msg_lgt+1
            elif resp[0] == 0xfc:
                msg_lgt = struct.unpack("!I", resp[1:5])[0]
                log.info("Node: {}".format(resp[1+4:1+4+msg_lgt].decode('utf-8')))
                this_ret['error'] = False
                skip_lgt = 1+4+msg_lgt+1
            else:
                this_ret["resp"] = True
                this_ret['error'] = False
            ret.append(this_ret)
            if len(resp) > skip_lgt:
                resp = resp[skip_lgt:]
            else:
                break
        return ret

def handle_nodes(title, node_specs):
    """
    Launches and monitors a set of nodes.
    Node-spec is [(node script, node count, n batches, [init msgs], [task msgs], [end msg]), ...]
    """
    node_ids = [[str(uuid.uuid4()) for _ in range(n[1])] for n in node_specs]
    nodes = [[subprocess.Popen([blackboard.config['performance']['python']] + [node_spec[0], u] + ([cfg_file] if cfg_file is not None else [])) for u in node_id] for node_spec, node_id in zip(node_specs, node_ids)]
    
    socks = []
    base_path = blackboard.TMP_PATH
    for node_id_list, node_spec_list in zip(node_ids, node_specs):
        socks.append([])
        for node_id in node_id_list:
            while True:
                try:
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sock.settimeout(0)
                    sock.connect(os.path.join(blackboard.config['data']['tmpdir'], "pepid_socket_{}".format(node_id)))
                    socks[-1].append(sock)
                    break
                except FileNotFoundError:
                    time.sleep(1)
                    continue

    n_total_nodes = sum([n[1] for n in node_specs])
    progress = tqdm.tqdm(desc=title+":INIT", total=n_total_nodes, mininterval=0)

    init_messages = {socks[node_type][node_count].getpeername() : node_specs[node_type][3][node_count] for node_type in range(len(node_specs)) for node_count in range(node_specs[node_type][1])}
    term_messages = {socks[node_type][node_count].getpeername() : node_specs[node_type][5][node_count] for node_type in range(len(node_specs)) for node_count in range(node_specs[node_type][1])}
    work_messages = {}
    for node_type in range(len(node_specs)):
        work_messages[node_type] = queue.SimpleQueue()
        for m in node_specs[node_type][4]:
            work_messages[node_type].put(m)

    all_socks = [sock for s in socks for sock in s]
    sock_to_type = {}
    for node_type in range(len(node_specs)):
        for sock in socks[node_type]:
            sock_to_type[sock.getpeername()] = node_type

    n_ready = 0
    done_nodes = []
    while n_ready < n_total_nodes:
        inp, out, errs = select.select(all_socks, all_socks, all_socks)
        for sock in out:
            if sock not in done_nodes:
                sock.sendall(init_messages[sock.getpeername()])
                done_nodes.append(sock)
        for sock in inp:
            if sock in done_nodes:
                try:
                    msg = sock.recv(1024)
                except socket.timeout:
                    continue
                successes = handle_response(msg)
                for success in successes:
                    if success['resp'] and success['code'] == 0xff:
                        continue # node init in progess
                    elif success['resp'] and success['code'] == 0xfe:
                        progress.update()
                        n_ready += 1 # node init successful
        for sock in errs:
            log.error("Broken node with invalid error return, quitting")
            sys.exit(2)
        
    progress.close()

    n_total_batches = sum([node_spec[2] for node_spec in node_specs])
    progress = tqdm.tqdm(desc=title+":RUN", total=n_total_batches, mininterval=0)

    waiting_socks = copy.copy(all_socks)
    engaged_socks = []
    running_socks = []
    done_batches = [0] * len(node_specs)
    while sum(done_batches) < n_total_batches:
        inp, out, errs = select.select(all_socks, all_socks, all_socks)
        for sock in out:
            if sock in waiting_socks:
                if not work_messages[sock_to_type[sock.getpeername()]].empty():
                    waiting_socks.remove(sock)
                    msg = work_messages[sock_to_type[sock.getpeername()]].get()
                    sock.sendall(msg)
                    engaged_socks.append(sock)
        for sock in inp:
            try:
                resp = sock.recv(1024)
            except socket.timeout:
                continue
            successes = handle_response(resp)
            for success in successes:
                if sock in engaged_socks:
                    if success['resp'] and success['code'] == 0xff:
                        engaged_socks.remove(sock)
                        running_socks.append(sock)
                elif sock in running_socks:
                    if success['resp'] and success['code'] == 0xfe:
                        running_socks.remove(sock)
                        waiting_socks.append(sock)
                        progress.update()
                        done_batches[sock_to_type[sock.getpeername()]] += 1
        for sock in errs:
            log.error("Broken node with invalid error return, quitting")
            sys.exit(2)

    progress.close()
    progress = tqdm.tqdm(desc=title+":TERM", total=n_total_nodes, mininterval=0)

    n_ready = 0
    done_nodes = []
    while n_ready < n_total_nodes:
        inp, out, errs = select.select(all_socks, all_socks, all_socks)
        for sock in out:
            if sock not in done_nodes:
                sock.sendall(term_messages[sock.getpeername()])
                done_nodes.append(sock)
        for sock in inp:
            try:
                msg = sock.recv(1024)
            except socket.timeout:
                continue
            successes = handle_response(msg)
            for success in successes:
                if success['resp'] and success['code'] == 0xff:
                    progress.update()
                    n_ready += 1
        for sock in errs:
            log.error("Broken node with invalid error return, quitting")
            sys.exit(2)

    progress.close()

    for node_list in nodes:
        for node in node_list:
            node.terminate()

def run():
    """
    Entry point
    """

    blackboard.setup_constants()

    import queries
    import db
    import processing
    import pepid_io
    import search

    try:
        log.info("Phases to run: | " + ("DB Processing | " if blackboard.config['pipeline'].getboolean('db processing') else "") +
                                    ("Postprocessing | " if blackboard.config['pipeline'].getboolean('postprocessing') else "") +
                                    ("Score | " if blackboard.config['pipeline'].getboolean('score') else "") +
                                    ("Search Postprocessing | " if blackboard.config['pipeline'].getboolean("postprocess search") else "") +
                                    ("Output CSV | " if blackboard.config['pipeline'].getboolean('output csv') else ""))
        log.info("Preparing Input Databases...")
        db_paths = [blackboard.DB_PATH + "_q.sqlite"]
        if blackboard.config['pipeline'].getboolean('db processing'):
            db_paths.append(blackboard.DB_PATH + "_cands.sqlite")
        if blackboard.config['pipeline'].getboolean('score'):
            db_paths.append(blackboard.DB_PATH + ".sqlite")
        for p in db_paths:
            if os.path.exists(p):
                os.remove(p)
        blackboard.prepare_connection()
        queries.prepare_db()
        if blackboard.config['pipeline'].getboolean('db processing'):
            db.prepare_db()
        blackboard.init_results_db()

        if blackboard.config['pipeline'].getboolean('score'):
            log.info("Preparing Input Processing Nodes...")
            
            qnodes = blackboard.config['performance'].getint('query nodes')
            dbnodes = blackboard.config['performance'].getint('db nodes')
            snodes = blackboard.config['performance'].getint('search nodes')

            if qnodes < 0 or dbnodes < 0 or snodes < 0:
                log.fatal("node settings are query={}, db={}, search={}, but only values 0 or above allowed.".format(qnodes, dbnodes, snodes))
                sys.exit(2)

            batch_size = blackboard.config['performance'].getint('batch size')
            n_db = db.count_db()
            n_queries = queries.count_queries()
            n_db_batches = math.ceil(n_db / batch_size)
            n_query_batches = math.ceil(n_queries / batch_size)
        
            base_path = blackboard.TMP_PATH
            qspec = [("queries_node.py", qnodes, n_query_batches,
                            [struct.pack("!cI{}sc".format(len(base_path)), bytes([0x00]), len(base_path), base_path.encode("utf-8"), "$".encode("utf-8")) for _ in range(qnodes)],
                            [struct.pack("!cQQc", bytes([0x01]), b * batch_size, min((b+1) * batch_size, n_queries), "$".encode("utf-8")) for b in range(n_query_batches)],
                            [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(qnodes)])]
            dbspec = [("db_node.py", dbnodes, n_db_batches,
                            [struct.pack("!cI{}sc".format(len(base_path)), bytes([0x00]), len(base_path), base_path.encode("utf-8"), "$".encode("utf-8")) for _ in range(dbnodes)],
                            [struct.pack("!cQQc", bytes([0x01]), b * batch_size, min((b+1) * batch_size, n_db), "$".encode("utf-8")) for b in range(n_db_batches)],
                            [struct.pack("!cc", bytes([0x7f]), "$".encode('utf-8')) for _ in range(dbnodes)])]

            handle_nodes("Input Processing", (qspec + dbspec) if blackboard.config['pipeline'].getboolean('db processing') else qspec)

        if blackboard.config['pipeline'].getboolean('postprocessing'):
            idx = 0

            qnodes = blackboard.config['performance'].getint('post query nodes')
            dbnodes = blackboard.config['performance'].getint('post db nodes')

            batch_start = 0
            n_db = db.count_peps()
            n_db_batches = math.ceil(n_db / batch_size)

            if qnodes < 0 or dbnodes < 0:
                log.fatal("post-processing node settings are query={}, db={}, but only values 0 or above allowed.".format(qnodes, dbnodes))
                sys.exit(2)

            qspec = [("queries_node.py", qnodes, n_query_batches,
                            [struct.pack("!cI{}sc".format(len(base_path)), bytes([0x00]), len(base_path), base_path.encode("utf-8"), "$".encode("utf-8")) for _ in range(qnodes)],
                            [struct.pack("!cQQc", bytes([0x02]), b * batch_size, min((b+1) * batch_size, n_queries), "$".encode("utf-8")) for b in range(n_query_batches)],
                            [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(qnodes)])]
            dbspec = [("db_node.py", dbnodes, n_db_batches,
                            [struct.pack("!cI{}sc".format(len(base_path)), bytes([0x00]), len(base_path), base_path.encode("utf-8"), "$".encode("utf-8")) for _ in range(dbnodes)],
                            [struct.pack("!cQQc", bytes([0x02]), b * batch_size, min((b+1) * batch_size, n_db), "$".encode('utf-8')) for b in range(n_db_batches)],
                            [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(dbnodes)])]

            handle_nodes("Input Postprocessing", qspec + dbspec)

        cur = blackboard.CONN.cursor()
        blackboard.execute(cur, "CREATE INDEX IF NOT EXISTS c.cand_mass_idx ON candidates (mass ASC);")
        del cur

        if blackboard.config['pipeline'].getboolean('score'):
            n_search_batches = math.ceil(n_queries / batch_size)
            sspec = [("search_node.py", snodes, n_search_batches,
                            [struct.pack("!cI{}sc".format(len(base_path)), bytes([0x00]), len(base_path), base_path.encode("utf-8"), "$".encode("utf-8")) for _ in range(snodes)],
                            [struct.pack("!cQQc", bytes([0x01]), b * batch_size, min((b+1) * batch_size, n_queries), "$".encode("utf-8")) for b in range(n_search_batches)],
                            [struct.pack("!cc", bytes([0x7f]), "$".encode("utf-8")) for _ in range(snodes)])]

            handle_nodes("Search", sspec)

            import glob
            fname_pattern = list(filter(lambda x: len(x) > 0, blackboard.config['data']['database'].split('/')))[-1].rsplit('.', 1)[0] + "_*_pepidpart.sqlite"
            fname_path = os.path.join(blackboard.TMP_PATH, fname_pattern)

            files = glob.glob(fname_path)

            cur = blackboard.CONN.cursor()
            blackboard.execute(cur, "CREATE INDEX IF NOT EXISTS res_score_idx ON results (title ASC, score DESC);")
            for f in tqdm.tqdm(files, total=len(files), desc="Merging Results"):
                blackboard.execute(cur, "ATTACH DATABASE ? AS results_part;", (f,))
                blackboard.execute(cur, "INSERT OR IGNORE INTO results SELECT * FROM results_part.results;")
                blackboard.commit()
                blackboard.execute(cur, "DETACH DATABASE results_part;")
                os.remove(f)
            del cur

        if blackboard.config['pipeline'].getboolean('postprocess search'):
            log.info("Search complete. Post-processing results...")
            processing.post_process()
        log.info("Done.")
        if blackboard.config['pipeline'].getboolean('output csv'):
            log.info("Saving results to {}.".format(blackboard.config['data']['output']))
            pepid_io.write_output()

    finally:
        log.info("Cleaning up...")
        if len(blackboard.config['data']['tmpdir']) > 0:
            os.system("rm -rf {}".format(os.path.join(blackboard.config['data']['tmpdir'], "pepid_socket*")))
            os.system("rm -rf {}".format(os.path.join(blackboard.config['data']['tmpdir'], "pepidtmp*")))
            # Note: final db not removed for future reuse

            if blackboard.LOCK is not None:
                blackboard.LOCK.close()
                os.system("rm -rf {}".format(os.path.join(blackboard.config['data']['tmpdir'], ".lock")))

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("USAGE: {} config.cfg".format(sys.argv[0]))
        sys.exit(1)

    global cfg_file
    global log

    blackboard.config.read('data/default.cfg')

    blackboard.config.read(sys.argv[1])
    cfg_file = sys.argv[1]

    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=eval("logging.{}".format(blackboard.config['logging']['level'].upper())))
    log = logging.getLogger("pepid")

    blackboard.TMP_PATH = tempfile.mkdtemp(prefix="pepidtmp_", dir=blackboard.config['data']['tmpdir'])
    if(not os.path.exists(blackboard.TMP_PATH)):
        os.mkdir(blackboard.TMP_PATH)

    blackboard.LOCK = open(os.path.join(blackboard.config['data']['tmpdir'], ".lock"), "wb")

    run()
