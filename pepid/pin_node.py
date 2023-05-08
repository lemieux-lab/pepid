import struct
import os
import time
import sys

if __package__ is None or __package__ == '':
    import node
    import blackboard
    import pepid_utils
else:
    from . import node
    from . import blackboard
    from . import pepid_utils

class PINNode(node.Node):
    def __init__(self, unix_sock):
        super().__init__(unix_sock)
        self.path = None
        self.file = None
        self.messages[0x00] = [None, self.prepare]
        self.messages[0x01] = ["!QQc", self.do]

    def do(self, start, end, _):
        if __package__ is None or __package__ == '':
            import pepid_percolator
        else:
            from . import pepid_percolator

        if not self.path:
            raise ValueError("'do' message received before 'prepare' message, aborting.")

        in_fname = blackboard.config['data']['output']
        fname, fext = in_fname.rsplit('.', 1)
        suffix = blackboard.config['rescoring']['suffix']
        pin_name = fname + suffix + "_pin.tsv"

        fin = open(in_fname, 'r')

        log_level = blackboard.config['logging']['level'].lower()

        header = next(fin).strip().split("\t")
        title_idx = header.index("title")

        lines = []
        cnt = -1
        prev = None
        for il, l in enumerate(fin):
            line = l.strip().split("\t")
            if line[title_idx] != prev:
                prev = line[title_idx]
                cnt += 1
                if cnt < start:
                    continue
                lines.append([])
                if cnt >= end:
                    break
            if start <= cnt < end:
                lines[-1].append(l.strip().split("\t"))

        fin.close()
     
        lines = pepid_utils.tsv_to_pin(header, lines, start)
        string = ""
        for l in lines:
            for ll in l:
                string += "\t".join(ll) + "\n"

        blackboard.lock()
        pin = open(pin_name, 'a')
        pin.write(string)
        pin.close()
        blackboard.unlock()

    def prepare(self, msg):
        lgt = struct.unpack("!I", msg[:4])[0]
        blackboard.TMP_PATH = struct.unpack("!{}sc".format(lgt), msg[4:])[0].decode('utf-8')
        self.path = blackboard.TMP_PATH
        blackboard.setup_constants()
        blackboard.LOCK = open(os.path.join(blackboard.TMP_PATH, ".lock"), "wb")
        blackboard.prepare_connection()

if __name__ == '__main__':
    node.init(PINNode)
