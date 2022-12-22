import uuid
import copy
import queue
import struct
import socket
import select
import os
import sys
import time
import tqdm

import blackboard

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

def handle_nodes(title, node_specs, tqdm_silence=False, cfg_file=None):
    """
    Launches and monitors a set of nodes.
    Node-spec is [(node script, node count, n batches, [init msgs], [task msgs], [end msg]), ...]
    """

    node_ids = [[str(uuid.uuid4()) for _ in range(n[1])] for n in node_specs]
    nodes = [[blackboard.subprocess([node_spec[0], u] + ([cfg_file] if cfg_file is not None else [])) for u in node_id] for node_spec, node_id in zip(node_specs, node_ids)]
    
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
    progress = tqdm.tqdm(desc=title+":INIT", total=n_total_nodes, mininterval=0, disable=tqdm_silence)

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
            sys.exit(-2)
        
    progress.close()

    n_total_batches = sum([node_spec[2] for node_spec in node_specs])
    progress = tqdm.tqdm(desc=title+":RUN", total=n_total_batches, mininterval=0, disable=tqdm_silence)

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
            sys.exit(-2)

    progress.close()
    progress = tqdm.tqdm(desc=title+":TERM", total=n_total_nodes, mininterval=0, disable=tqdm_silence)

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
            sys.exit(-2)

    progress.close()

    for node_list in nodes:
        for node in node_list:
            node.terminate()


