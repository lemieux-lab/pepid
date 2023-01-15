import traceback
import socket
import time
import random
import struct
import os
import uuid
import numpy
import tempfile

if __package__ is None or __package__ == '':
    import blackboard
else:
    from . import blackboard

# Messages:
# 0xff: acknowledged, followed by message code being acknowledged
# 0xfe: done, followed by code of task done
# 0xfd: error (next 4 bytes are size of incoming error message, followed by message as utf-8)
# 0xfc: message (e.g. logging, debug, etc.; works similar to error)
# 0x7f: quit

class Node():
    def __init__(self, unix_sock):
        self.unix_sock = unix_sock
        self.messages = {}

    def start(self):
        sock = socket.socket(family=socket.AF_UNIX, type=socket.SOCK_STREAM)
        sock.bind(self.unix_sock)
        sock.listen(1)
        conn, addr = sock.accept()
        while True:
            msg = conn.recv(1024)
            if not msg:
                continue
            if msg[0] == 0x7f:
                conn.sendall(struct.pack("!ccc", bytes([0xff]), bytes([msg[0]]), "$".encode('utf-8')))
                break
            else:
                if msg[0] in self.messages.keys():
                    conn.sendall(struct.pack("!ccc", bytes([0xff]), bytes([msg[0]]), "$".encode('utf-8')))
                    try:
                        if self.messages[msg[0]][0] is not None:
                            args = struct.unpack(self.messages[msg[0]][0], msg[1:])
                            self.messages[msg[0]][1](*list(args))
                        else:
                            self.messages[msg[0]][1](msg[1:])
                        conn.sendall(struct.pack("!ccc", bytes([0xfe]), bytes([msg[0]]), "$".encode('utf-8')))
                    except Exception as e:
                        err = "Failed to handle message {} -- {}".format(msg[0], repr(e)) 
                        conn.sendall(struct.pack("!cI{}sc".format(len(err)), bytes([0xfd]), len(err), err.encode('utf-8'), "$".encode('utf-8')))
                        if blackboard.config['logging']['level'].lower() == 'debug':
                            import traceback
                            traceback.print_tb(e.__traceback__)
                else:
                    err = "Unknown message code received: {}".format(msg[0])
                    conn.sendall(struct.pack("!cI{}s".format(len(err)), bytes([0xfd]), len(err), err.encode('utf-8')))

        blackboard.CONN.close()
        conn.close()
        sock.close()

def init(klass):
    import sys
    sock = str(uuid.uuid4())
    if len(sys.argv) > 1:
        sock = sys.argv[1]

    blackboard.config.read(blackboard.here("data/default.cfg"))

    if len(sys.argv) == 3:
        blackboard.config.read(sys.argv[2])

    this = klass(os.path.join(blackboard.config['data']['tmpdir'], "pepid_socket_" + sock))
    this.start()
