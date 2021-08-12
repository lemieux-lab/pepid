import blackboard
import struct
import os
import node
import time

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

        #import profile
        #profile.run("import search; search.search_core({}, {})".format(start, end))
        search.search_core(start, end)

    def prepare(self, msg):
        lgt = struct.unpack("!I", msg[:4])[0]
        self.path = struct.unpack("!{}sc".format(lgt), msg[4:])[0].decode('utf-8')

if __name__ == '__main__':
    node.init(SearchNode)
