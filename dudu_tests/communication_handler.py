from enum import Enum
import json
import os
import signal
import os, requests
import sys, traceback
import pickle
import shutil
import signal
import time, shutil


last_active_key = None
GET_DATAFRAME_JOB = "dataframe"
EVALUATE_MAPPING_JOB = "job"
base_url = os.environ['SERVICE_URL']
urls = {
    'get_job': base_url + "/getjob",
    'set_job': base_url + "/setjob",
    'return_key': base_url + "/return_key",
    'maxreward': base_url + "/maxreward",
    'evaluate': base_url + "/evaluate",
    'resetoldjobs': base_url + "/job/resetold"
}


def prRed(prt): print("\033[91m {}\033[00m".format(prt), flush=True)


def prGreen(prt): print("\033[92m {}\033[00m".format(prt), flush=True)


def prYellow(prt): print("\033[93m {}\033[00m".format(prt), flush=True)


if 'JOBTIMEOUT_SECONDS' in os.environ:
    jobtimeout = int(os.environ['JOBTIMEOUT_SECONDS'])
elif 'JOBTIMEOUT_MINUTES' in os.environ:
    jobtimeout = int(os.environ['JOBTIMEOUT_MINUTES']) * 60
else:
    jobtimeout = 4000

if 'CODEID' in os.environ:
    codeid = os.environ['CODEID']
else:
    codeid = 'user_provided_identifier_of_the_current_code'


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("Exit signal ", signum, " caught. Last active key =", last_active_key, flush=True)
        if last_active_key:
            requests.request("GET", urls['return_key'] + f"?key={last_active_key}", headers={}, data={})

        self.kill_now = True


def parse_cmdargs(wlparams: dict):
    """
    todo delete this?
    """
    try:
        cmdargs = wlparams['params']['cmdargs']
        import re
        _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
        cmdargs = _RE_COMBINE_WHITESPACE.sub(" ", cmdargs).strip()
    except:
        prRed("**** Must have at least project_path arguments *****")
        return None, None, -1, {
            'msg': "cmdargs of SPARTA jobs must include the '--project_path' argument in the 'cmdargs' field"}

    cmdargs_splited = cmdargs.split(" ")
    cmdargs = " ".join(cmdargs_splited)
    return cmdargs, None, None


def reset_old_jobs():
    req = requests.request("GET", urls['resetoldjobs'] + f"?codeid={codeid}&timeout={jobtimeout}", headers={}, data={})


def create_reward(latency, baseline_latency, batch_size, tweaks: int, ffproc_time_sec: float) -> str:
    """
    creates a json reward to autoML
    todo add further documentation here
    """
    reward = -1 * tweaks
    if tweaks == 0:
        reward = baseline_latency/latency

    jv = {
        "reward": reward,
        "rc": 0,
        "done": 1,
        "msg": "flexflow evaluation complete",
        "meta_data": {
            "IPS": float((1000 * batch_size)/latency),
            "reward": float(reward),
            "speedup": float(100*baseline_latency/latency),
            "latency": int(latency)
        }
    }
    return json.dumps(jv)
