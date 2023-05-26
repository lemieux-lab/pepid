import sys
import time
import os

if __package__ is None or __package__ == '':
    import blackboard
    from blackboard import here
else:
    from . import blackboard
    from .blackboard import here

def run(cfg):
    ret = 0

    try:
        blackboard.config.read(blackboard.here("data/default.cfg"))
        blackboard.config.read(cfg)
        blackboard.setup_constants()

        log_level = blackboard.config['logging']['level'].lower()

        if blackboard.config['pipeline'].getboolean('search'):
            proc = blackboard.subprocess([here("pepid_search.py"), cfg])
            while True:
                ret = proc.poll()
                if ret is not None:
                    break
                time.sleep(1) 

            if ret < 0:
                if log_level == 'debug':
                    sys.stderr.write("Terminated with error {}\n".format(ret))
                raise Exception(ret)

        if blackboard.config['pipeline'].getboolean('postsearch'):
            proc = blackboard.subprocess([here("pepid_postprocess.py"), cfg])
            while True:
                ret = proc.poll()
                if ret is not None:
                    break
                time.sleep(1)

            if ret < 0:
                if log_level == 'debug':
                    sys.stderr.write("Terminated with error {}\n".format(ret))
                raise Exception(ret)

        if blackboard.config['pipeline'].getboolean('output'):
            proc = blackboard.subprocess([here("pepid_io.py"), cfg])
            while True:
                ret = proc.poll()
                if ret is not None:
                    break
                time.sleep(1) 

            if ret < 0:
                if log_level == 'debug':
                    sys.stderr.write("Terminated with error {}\n".format(ret))
                raise Exception(ret)

        if blackboard.config['pipeline'].getboolean('report'):
            report_name = "gen_fdr_report.py"
            proc = blackboard.subprocess([here(report_name), cfg, "output"])
            while True:
                ret = proc.poll()
                if ret is not None:
                    break
                time.sleep(1) 

            if ret < 0:
                if log_level == 'debug':
                    sys.stderr.write("Terminated with error {}\n".format(ret))
                raise Exception(ret)

        if blackboard.config['pipeline'].getboolean('rescoring'):
            rescorer = blackboard.config['rescoring']['function']
            proc = blackboard.subprocess([here("pepid_rescore.py"), cfg])
            while True:
                ret = proc.poll()
                if ret is not None:
                    break
                time.sleep(1) 

            if ret < 0:
                if log_level == 'debug':
                    sys.stderr.write("Terminated with error {}\n".format(ret))
                raise Exception(ret)

        if blackboard.config['pipeline'].getboolean('rescoring report'):
            report_name = "gen_fdr_report.py"
            proc = blackboard.subprocess([here(report_name), cfg, "rescored"])
            while True:
                ret = proc.poll()
                if ret is not None:
                    break
                time.sleep(1) 

            if ret < 0:
                if log_level == 'debug':
                    sys.stderr.write("Terminated with error {}\n".format(ret))
                raise Exception(ret)

    finally:
        if log_level in ['debug', 'info']:
            sys.stderr.write("Cleaning up...\n")
        import os
        os.system("rm -rf {}".format(os.path.join(blackboard.config['data']['tmpdir'], "pepid_socket*")))
        os.system("rm -rf {}".format(os.path.join(blackboard.config['data']['workdir'], ".lock")))
        os.system("rm -rf {}".format(os.path.join(blackboard.config['data']['workdir'], ".lock_*"))) # Locks possibly created for e.g. devices

        sys.exit(ret)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: {} config.cfg".format(sys.argv[0]))
        sys.exit(-1)

    run(sys.argv[1])
