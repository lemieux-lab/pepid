import struct
import os
import time

if __package__ is None or __package__ == '':
    import blackboard
    import node
else:
    from . import blackboard
    from . import node

class QueryNode(node.Node):
    def __init__(self, unix_sock):
        super().__init__(unix_sock)
        self.path = None
        self.messages[0x00] = [None, self.prepare]
        self.messages[0x01] = ["!QQc", self.do]
        self.messages[0x02] = ["!QQc", self.do_post]

    def do(self, start, end, _):
        if __package__ is None or __package__ == '':
            import queries
        else:
            from . import queries

        if not self.path:
            raise ValueError("'do' message received before 'prepare' message, aborting.")

        queries.fill_queries(start, end)
        blackboard.CONN.commit()

    def do_post(self, start, end, _):
        if __package__ is None or __package__ == '':
            import queries
        else:
            from . import queries

        if not self.path:
            raise ValueError("'do_post' message received before 'prepare' message, aborting.")

        queries.user_processing(start, end)
        blackboard.CONN.commit()

    def prepare(self, msg):
        lgt = struct.unpack("!I", msg[:4])[0]
        self.path = struct.unpack("!{}sc".format(lgt), msg[4:])[0].decode('utf-8')
        blackboard.TMP_PATH = self.path
        blackboard.setup_constants()
        blackboard.LOCK = blackboard.acquire_lock()
        blackboard.prepare_connection()

if __name__ == '__main__':
    node.init(QueryNode)
