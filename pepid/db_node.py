import struct
import os
import time

if __package__ is None or __package__ == '':
    import blackboard
    import node
else:
    from . import blackboard
    from . import node

class DbNode(node.Node):
    def __init__(self, unix_sock):
        super().__init__(unix_sock)
        self.path = None
        self.messages[0x00] = [None, self.prepare]
        self.messages[0x01] = [None, self.do]
        self.messages[0x02] = ["!QQc", self.do_post]

    def do(self, msg):
        start, end = struct.unpack("!QQ", msg[:16])
        if __package__ is None or __package__ == '':
            import db
        else:
            from . import db
        if not self.path:
            raise ValueError("'do' message received before 'prepare' message, aborting.")

        # leftover msg = decoy or normal
        lgt = struct.unpack("!I", msg[16:20])[0]
        db.fill_db(start, end, struct.unpack("!{}sc".format(lgt), msg[20:])[0].decode('utf-8'))
        blackboard.CONN.commit()

    def do_post(self, start, end, _):
        if __package__ is None or __package__ == '':
            import db
        else:
            from . import db
        db.user_processing(start, end)
        blackboard.CONN.commit()

    def prepare(self, msg):
        lgt = struct.unpack("!I", msg[:4])[0]
        self.path = struct.unpack("!{}sc".format(lgt), msg[4:])[0].decode('utf-8')
        blackboard.TMP_PATH = self.path
        blackboard.setup_constants()
        blackboard.LOCK = open(os.path.join(blackboard.TMP_PATH, ".lock"), "wb")
        blackboard.prepare_connection()

if __name__ == '__main__':
    node.init(DbNode)
