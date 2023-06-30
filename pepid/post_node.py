import struct
import os
import time

if __package__ is None or __package__ == '':
    from pepid import blackboard
    from pepid import node
    from pepid import pepid_utils
else:
    from . import blackboard
    from . import node
    from . import pepid_utils

class PostNode(node.Node):
    def __init__(self, unix_sock):
        super().__init__(unix_sock)
        self.path = None
        self.messages[0x00] = [None, self.prepare]
        self.messages[0x01] = [None, self.do]

    def do(self, msg):
        start, end = struct.unpack("!QQ", msg[:16])

        post_fn = pepid_utils.import_or(blackboard.config['postsearch']['function'], None)

        if not self.path:
            raise ValueError("'do' message received before 'prepare' message, aborting.")

        if post_fn is not None:
            post_fn(start, end)
            blackboard.CONN.commit()
        else:
            blackboard.LOG.warning("Could not find postprocessing function '{}', not applying postprocessing".format(blackboard.config['postsearch']['function']))
            return

    def prepare(self, msg):
        lgt = struct.unpack("!I", msg[:4])[0]
        self.path = struct.unpack("!{}sc".format(lgt), msg[4:])[0].decode('utf-8')
        blackboard.TMP_PATH = self.path
        blackboard.setup_constants()
        blackboard.LOCK = blackboard.acquire_lock()
        blackboard.prepare_connection()

if __name__ == '__main__':
    node.init(PostNode)
