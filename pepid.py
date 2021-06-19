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

    if blackboard.config['pipeline'].getboolean('search'):
        proc = subprocess.Popen([blackboard.config['performance']['python'], "pepid_search.py", sys.argv[1]])
        while proc.poll() is None:
            time.sleep(1)

    if blackboard.config['pipeline'].getboolean('report'):
        report_name = "gen_{}_report.py".format(blackboard.config['pipeline']['report type'])
        proc = subprocess.Popen([blackboard.config['performance']['python'], report_name, sys.argv[1], "output"])
        while proc.poll() is None:
            time.sleep(1)

    if blackboard.config['pipeline'].getboolean('rescore'):
        rescorer = blackboard.config['pipeline']['rescorer']
        proc = subprocess.Popen([blackboard.config['performance']['python'], "pepid_rescore.py", sys.argv[1]])
        while proc.poll() is None:
            time.sleep(1) 

    if blackboard.config['pipeline'].getboolean('final report'):
        report_name = "gen_{}_report.py".format(blackboard.config['pipeline']['final report type'])
        proc = subprocess.Popen([blackboard.config['performance']['python'], report_name, sys.argv[1], "rescored"])
        while proc.poll() is None:
            time.sleep(1)
