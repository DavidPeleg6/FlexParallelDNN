import time, os
import os, requests
import sys, traceback
import pickle
import shutil
import signal
import time, shutil

last_active_key = None
GET_DATAFRAME_JOB = "dataframe"
EVALUATE_MAPPING_JOB = "job"


def prRed(prt): print("\033[91m {}\033[00m".format(prt), flush=True)


def prGreen(prt): print("\033[92m {}\033[00m".format(prt), flush=True)


def prYellow(prt): print("\033[93m {}\033[00m".format(prt), flush=True)


print("Worker started")
base_url = os.environ['SERVICE_URL']

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

urls = {
    'get_job': base_url + "/getjob",
    'set_job': base_url + "/setjob",
    'return_key': base_url + "/return_key",
    'maxreward': base_url + "/maxreward",
    'evaluate': base_url + "/evaluate",
    'resetoldjobs': base_url + "/job/resetold"
}


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


def get_data_frame(params, tmp_project_folder, cmdargs):
    # params = params['wlparams']
    prYellow("worker::get_data_frame")
    workload_fpath = params['wlparams']['workload_fpath']

    # This methos must  return a dictionary with these elements included:
    # dst_op: a dictionary of sources and destinations for each layer.
    # features: a dictionary of features for each layer containing one hot encoding of operator type from the list of operators, and the volume of operators (input and output volumes). The user will need to build a one hot encoding table for all the operators in the network.
    # action_space: the parameters to be optimized by the AutoML service. should be a dictionary of parameter name and the available options to tweak said parameter. Each parameter should also hold a key for whether the tweak options are discrete or not.

    # Example:
    # dataframe = {"action_space":{"action_type":{"device_start_idx":[["\"0\"","\"1\""],"discrete"],"strategy":[["1,1,1,1","1,1,1,2","1,2,1,1","2,1,1,1"],"discrete"]}},"dst_op":{"Conv2D_100":["Pool2D_101"],"Conv2D_102":["Pool2D_103"],"Conv2D_104":["Conv2D_105"],"Conv2D_105":["Conv2D_106"],"Conv2D_106":["Pool2D_107"],"Dense_109":["Dense_110"],"Dense_110":["Dense_111"],"Dense_111":["Softmax_112"],"Flat_108":["Dense_109"],"Pool2D_101":["Conv2D_102"],"Pool2D_103":["Conv2D_104"],"Pool2D_107":["Flat_108"],"Softmax_112":[]},"features":{"Conv2D_100":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23296,0,51380224],"Conv2D_102":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,307392,11943936,35831808],"Conv2D_104":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,663936,8306688,16613376],"Conv2D_105":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,884992,16613376,11075584],"Conv2D_106":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,590080,11075584,11075584],"Dense_109":[0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,37752832,2359296,1048576],"Dense_110":[0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16781312,1048576,1048576],"Dense_111":[0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,40970,1048576,2560],"Flat_108":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,,0,0,0,0,2359296,2359296],"Pool2D_101":[0,0,0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51380224,11943936],"Pool2D_103":[0,0,0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35831808,8306688],"Pool2D_107":[0,0,0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11075584,2359296],"Softmax_112":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,,0,0,0,2560,2560]},"strategy":{}}
    return {'normalized_state_df': dataframe, 'rc': 0}


def evaluate(params, tmp_project_folder, cmdargs, NAT_parameter_dict, version="v1"):
    workload_fpath = params['wlparams']['workload_fpath']
    options = params.get("options", None)
    mapping = params['mapping']
    # Example of a "mapping" instance:
    # mapping = {"Conv2D_100":{"device_start_idx":0,"param1":1,"param2":1,"param3":1,"param4":1},"Pool2D_107":{"device_start_idx":0,"param1":1,"param2":1,"param3":1,"param4":4},"Softmax_112":{"device_start_idx":0,"param1":1,"param2":4,"param3":1,"param4":1}}

    # Evaluate
    # DOMAIN SPECIFIC CODEID
    # =========
    # Example data for illustration purposes
    meta_data = {"IPS": 300.2, "Latency": 3.331}
    baselineIPS = 350.0
    processing_time = 10.3
    rwd = 300.2 / baselineIPS
    # =========
    rc_dict = {'rc': 0, 'done': 1, 'reward': rwd, 'processing_time': processing_time,
               'msg': 'Evaluate completed', 'meta_data': meta_data}
    return rc_dict


def reset_old_jobs():
    req = requests.request("GET", urls['resetoldjobs'] + f"?codeid={codeid}&timeout={jobtimeout}", headers={}, data={})


if __name__ == "__main__":
    # os.remove(constant.DATAFRAME_CACHE_FILE)
    print(len(sys.argv))
    print(sys.argv)
    print("SPAAS worker started 2")
    # if len(sys.argv)>=2:
    #    if sys.argv[1] == 1:
    #        sparta_measurements.reset_old_jobs()
    # from pathlib import Path
    # from sparta_subprocess import sparta_measurements
    killer = GracefulKiller()
    from tempfile import NamedTemporaryFile, TemporaryDirectory

    os.makedirs("/tmp_projects/projects/", exist_ok=True)
    t = TemporaryDirectory(dir='/tmp_projects/projects/', prefix='tmp')
    tmpfolder = t.name
    tmpfolder_graph = tmpfolder + "_graph"
    print("project dirs: ", tmpfolder, tmpfolder_graph)
    os.makedirs(tmpfolder_graph, exist_ok=True)
    try:
        while not killer.kill_now:
            with open('/tmp/healthy', 'w') as f:
                pass
            remaining_jobs = 0
            attempts = 5
            while attempts > 0:  # Try 5 times, because the service might not yet be available.
                print("Get job from: " + urls['get_job'] + f"?codeid={codeid}")
                try:
                    req = requests.request("GET", urls['get_job'] + f"?codeid={codeid}", headers={}, data={})
                except Exception as e:
                    t = 3
                    print(f"Exception {e}. Trying again after {t} seconds")
                    time.sleep(t)
                    continue

                # if not req.status_code==404 and not req.status_code==502 and not req.status_code==503 :
                if req.status_code == 200:
                    break
                if not req.status_code == 200:
                    print("req.status_code=", req.status_code, flush=True)
                    time.sleep(10)
                    continue
                time.sleep(3)
                # print("Failed calling "+ urls['get_job']+ " Trying "+str(attempts)+" more times",flush=True)
                # attempts -=1
            if attempts == 0:
                raise Exception("EGRL service not available. Failed calling: " + urls['get_job'])
            prGreen(f"req.status_code={req.status_code}")
            # print("req.text=", req.text,flush=True)
            req = req.json()
            # print(req,flush=True)
            if req['key']:
                prYellow("hash=" + str(req['key']))
                last_active_key = req['key']

                for v in [tmpfolder, tmpfolder_graph]:
                    for root, dirs, files in os.walk(v):
                        for f in files:
                            os.unlink(os.path.join(root, f))
                        for d in dirs:
                            shutil.rmtree(os.path.join(root, d))

                try:
                    # ===========
                    # Pre-process the ticket.
                    # ===========
                    job_definition = req['job']
                    t = time.time()
                    cmdargs, error_code, error_msg = parse_cmdargs(job_definition['wlparams'])
                    job_definition['wlparams']['params']['cmdargs'] = cmdargs
                    if error_code:
                        print(f"Exception in evaluating job: {error_msg['msg']}")
                        requests.request("GET", urls['return_key'] + f"?key={last_active_key}", headers={}, data={})
                        continue

                    # ===========
                    # Process tickets from service:
                    # ===========
                    if GET_DATAFRAME_JOB in last_active_key:
                        # Get DataFrame
                        results_json = get_data_frame(job_definition, tmpfolder, cmdargs)
                        # Example of a results_json instance:
                        # results_json = {"normalized_state_df":{"action_space":{"action_type":{"device_start_idx":[["\"0\"","\"1\""],"discrete"],"strategy":[["1,1,1,1","1,1,1,2","1,2,1,1","2,1,1,1"],"discrete"]}},"dst_op":{"Conv2D_100":["Pool2D_101"],"Conv2D_102":["Pool2D_103"],"Conv2D_104":["Conv2D_105"],"Conv2D_105":["Conv2D_106"],"Conv2D_106":["Pool2D_107"],"Dense_109":["Dense_110"],"Dense_110":["Dense_111"],"Dense_111":["Softmax_112"],"Flat_108":["Dense_109"],"Pool2D_101":["Conv2D_102"],"Pool2D_103":["Conv2D_104"],"Pool2D_107":["Flat_108"],"Softmax_112":[]},"features":{"Conv2D_100":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23296,0,51380224],"Conv2D_102":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,307392,11943936,35831808],"Conv2D_104":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,663936,8306688,16613376],"Conv2D_105":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,884992,16613376,11075584],"Conv2D_106":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,590080,11075584,11075584],"Dense_109":[0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,37752832,2359296,1048576],"Dense_110":[0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16781312,1048576,1048576],"Dense_111":[0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,40970,1048576,2560],"Flat_108":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,,0,0,0,0,2359296,2359296],"Pool2D_101":[0,0,0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51380224,11943936],"Pool2D_103":[0,0,0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35831808,8306688],"Pool2D_107":[0,0,0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11075584,2359296],"Softmax_112":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,,0,0,0,2560,2560]},"strategy":{}}}
                    elif EVALUATE_MAPPING_JOB in last_active_key:
                        # Evaluate
                        # ========
                        # Example of a "job_definition"
                        # job_definition = {"issued_time":1625312390.1763651,"jobtype":"job","mapping":{"Conv2D_100":{"device_start_idx":0,"param1":1,"param2":1,"param3":1,"param4":1},"Pool2D_107":{"device_start_idx":0,"param1":1,"param2":1,"param3":1,"param4":4},"Softmax_112":{"device_start_idx":0,"param1":1,"param2":4,"param3":1,"param4":1}}}
                        # ========
                        results_json = evaluate(job_definition, tmpfolder, cmdargs)
                        reward = results_json['reward']
                    proc_time = time.time() - t

                    # ===========
                    # Report back results to service
                    # ===========
                    r = requests.request("POST", urls['set_job'] + f"?key={last_active_key}", headers={},
                                         json=results_json)
                    last_active_key = None
                except Exception as e:
                    print("Exception in evaluating job, ", e)
                    requests.request("GET", urls['return_key'] + f"?key={last_active_key}", headers={}, data={})

            # os.remove('/tmp/healthy')
            if req['remaining_jobs'] == 0:
                reset_old_jobs()
                time.sleep(5)

    except Exception as e:
        print("Exception: ", e)

        if last_active_key:
            print("Goodbye in Exception A", flush=True)
            requests.request("GET", urls['return_key'] + f"?key={last_active_key}", headers={}, data={})
        print("-" * 60)
        traceback.print_exc(file=sys.stdout)
        print("-" * 60)
    print("Goodbye EOP", flush=True)

