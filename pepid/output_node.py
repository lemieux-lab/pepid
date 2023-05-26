import struct
import os
import time
import sys

if __package__ is None or __package__ == '':
    import node
    import blackboard
else:
    from . import node
    from . import blackboard

class OutputNode(node.Node):
    def __init__(self, unix_sock):
        super().__init__(unix_sock)
        self.path = None
        self.file = None
        self.messages[0x00] = [None, self.prepare]
        self.messages[0x01] = ["!QQc", self.do]

    def do(self, start, end, _):
        if __package__ is None or __package__ == '':
            import pepid_io
        else:
            from . import pepid_io

        if not self.path:
            raise ValueError("'do' message received before 'prepare' message, aborting.")

        pepid_io.write_output(start, end)

    def prepare(self, msg):
        lgt = struct.unpack("!I", msg[:4])[0]
        blackboard.TMP_PATH = struct.unpack("!{}sc".format(lgt), msg[4:])[0].decode('utf-8')
        self.path = blackboard.TMP_PATH
        blackboard.setup_constants()
        blackboard.LOCK = blackboard.acquire_lock()
        blackboard.prepare_connection()

if __name__ == '__main__':
    node.init(OutputNode)
