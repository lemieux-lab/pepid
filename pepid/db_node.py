import struct
import os
import blackboard
import time

import node

class DbNode(node.Node):
    def __init__(self, unix_sock):
        super().__init__(unix_sock)
        self.path = None
        self.messages[0x00] = [None, self.prepare]
        self.messages[0x01] = ["!QQc", self.do]
        self.messages[0x02] = ["!QQc", self.do_post]

    def do(self, start, end, _):
        import db
        if not self.path:
            raise ValueError("'do' message received before 'prepare' message, aborting.")

        db.fill_db(start, end)
        blackboard.CONN.commit()

    def do_post(self, start, end, _):
        import db
        db.user_processing(start, end)
        blackboard.CONN.commit()

    def prepare(self, msg):
        lgt = struct.unpack("!I", msg[:4])[0]
        self.path = struct.unpack("!{}sc".format(lgt), msg[4:])[0].decode('utf-8')

if __name__ == '__main__':
    node.init(DbNode)