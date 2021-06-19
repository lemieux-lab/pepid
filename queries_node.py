import blackboard
import struct
import os
import psycopg2
import time

import queries
import node

class QueryNode(node.Node):
    def __init__(self, unix_sock):
        super().__init__(unix_sock)
        self.path = None
        self.messages[0x00] = [None, self.prepare]
        self.messages[0x01] = ["!QQc", self.do]
        self.messages[0x02] = ["!QQc", self.do_post]

    def do(self, start, end, _):
        if not self.path:
            raise ValueError("'do' message received before 'prepare' message, aborting.")

        done = False
        conn = None
        while not done:
            try:
                conn = psycopg2.connect(host="localhost", port='9991', database="postgres")
                queries.fill_queries(conn, start, end)
                done = True
            except psycopg2.DatabaseError as e:
                if conn is not None:
                    conn.rollback()
                    conn.close()
                    conn = None
                raise e
            finally:
                if conn is not None:
                    conn.commit()
                    conn.close()

    def do_post(self, start, end, _):
        if not self.path:
            raise ValueError("'do_post' message received before 'prepare' message, aborting.")

        done = False
        conn = None
        while not done:
            try:
                conn = psycopg2.connect(host="localhost", database="postgres")
                queries.user_processing(conn, start, end)
                done = True
            except psycopg2.DatabaseError as e:
                if conn is not None:
                    conn.rollback()
                    conn.close()
                    conn = None
                raise e
            finally:
                if conn is not None:
                    conn.commit()
                    conn.close()

    def prepare(self, msg):
        lgt = struct.unpack("!I", msg[:4])[0]
        self.path = struct.unpack("!{}sc".format(lgt), msg[4:])[0].decode('utf-8')

if __name__ == '__main__':
    node.init(QueryNode)
