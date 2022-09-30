import blackboard
import struct
import os
import node
import time
import sys

class SearchNode(node.Node):
    def __init__(self, unix_sock):
        super().__init__(unix_sock)
        self.path = None
        self.messages[0x00] = [None, self.prepare]
        self.messages[0x01] = ["!QQc", self.do]

    def do(self, start, end, _):
        import search

        if not self.path:
            raise ValueError("'do' message received before 'prepare' message, aborting.")

        search.search_core(start, end)

    def prepare(self, msg):
        lgt = struct.unpack("!I", msg[:4])[0]
        blackboard.TMP_PATH = struct.unpack("!{}sc".format(lgt), msg[4:])[0].decode('utf-8')
        self.path = blackboard.TMP_PATH
        blackboard.setup_constants()
        blackboard.LOCK = open(os.path.join(blackboard.TMP_PATH, ".lock"), "wb")
        blackboard.init_results_db(True, base_dir=blackboard.TMP_PATH)
        blackboard.prepare_connection()

if __name__ == '__main__':
    node.init(SearchNode)
