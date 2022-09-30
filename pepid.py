import sys
import blackboard
import subprocess
import time

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: {} config.cfg".format(sys.argv[0]))
        sys.exit(1)

    blackboard.config.read("data/default.cfg")
    blackboard.config.read(sys.argv[1])

    log_level = blackboard.config['logging']['level'].lower()

    if blackboard.config['pipeline'].getboolean('search'):
        proc = subprocess.Popen([blackboard.config['performance']['python'], "pepid_search.py", sys.argv[1]])
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            time.sleep(1) 

        if ret < 0:
            if log_level == 'debug':
                sys.stderr.write("Terminated with error {}\n".format(ret))
            sys.exit(ret)

    if blackboard.config['pipeline'].getboolean('report'):
        report_name = "gen_{}_report.py".format(blackboard.config['report']['type'])
        proc = subprocess.Popen([blackboard.config['performance']['python'], report_name, sys.argv[1], "output"])
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            time.sleep(1) 

        if ret < 0:
            if log_level == 'debug':
                sys.stderr.write("Terminated with error {}\n".format(ret))
            sys.exit(ret)

    if blackboard.config['pipeline'].getboolean('rescoring'):
        rescorer = blackboard.config['rescoring']['function']
        proc = subprocess.Popen([blackboard.config['performance']['python'], "pepid_rescore.py", sys.argv[1]])
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            time.sleep(1) 

        if ret < 0:
            if log_level == 'debug':
                sys.stderr.write("Terminated with error {}\n".format(ret))
            sys.exit(ret)

    if blackboard.config['pipeline'].getboolean('rescoring report'):
        report_name = "gen_{}_report.py".format(blackboard.config['report']['type'])
        proc = subprocess.Popen([blackboard.config['performance']['python'], report_name, sys.argv[1], "rescored"])
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            time.sleep(1) 

        if ret < 0:
            if log_level == 'debug':
                sys.stderr.write("Terminated with error {}\n".format(ret))
            sys.exit(ret)
