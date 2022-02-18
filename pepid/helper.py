import blackboard
import fcntl

def lock():
    fcntl.lockf(blackboard.LOCK, fcntl.LOCK_EX)

def unlock():
    fcntl.lockf(blackboard.LOCK, fcntl.LOCK_UN)
