import copy
from dudu_tests.Optimizer import simulate, wrap_model
from dudu_tests.layerwise_data_parallel import _gather
import torch
import torch.distributed as dist
from dudu_tests import Optimizer, strategy_handler
from dudu_tests.strategy_struct import LayerStrategy
import json
import os, requests
import sys, traceback
import signal
import time, shutil
from datetime import timedelta

STRATEGY_SEEN_KEY = 'strategy_seen'
STRATEGY_KEY = 'strategy'
last_active_key = None
GET_DATAFRAME_JOB = "dataframe"
EVALUATE_MAPPING_JOB = "job"
base_url = os.environ['SERVICE_URL']
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


def prRed(prt): print("\033[91m {}\033[00m".format(prt), flush=True)


def prGreen(prt): print("\033[92m {}\033[00m".format(prt), flush=True)


def prYellow(prt): print("\033[93m {}\033[00m".format(prt), flush=True)


if 'JOBTIMEOUT_SECONDS' in os.environ:
    jobtimeout = int(os.environ['JOBTIMEOUT_SECONDS'])
elif 'JOBTIMEOUT_MINUTES' in os.environ:
    jobtimeout = int(os.environ['JOBTIMEOUT_MINUTES']) * 60
else:
    jobtimeout = 40


class GracefulKiller:
    kill_now = False

    def __init__(self):
        # signal.signal(signal.SIGINT, self.exit_gracefully)
        # signal.signal(signal.SIGTERM, self.exit_gracefully)
        # todo fix this so that it will catch and still manage to close all processes
        pass

    def exit_gracefully(self, signum, frame):
        print("Exit signal ", signum, " caught. Last active key =", last_active_key, flush=True)
        # if dist.get_rank() == 0:
        #     if last_active_key:
        #         requests.request("GET", urls['return_key'] + f"?key={last_active_key}", headers={}, data={})
        # os.kill(os.getpid(), signal.SIGTERM)
        dist.destroy_process_group()
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


def get_data_frame(params, tmp_project_folder):
    # params = params['wlparams']
    prYellow("worker::get_data_frame")
    # workload_fpath = params['wlparams']['workload_fpath']

    # This method must  return a dictionary with these elements included:
    # dst_op: a dictionary of sources and destinations for each layer.
    # features: a dictionary of features for each layer containing one hot encoding of operator type from the list of operators, and the volume of operators (weights, input and output volumes). The user will need to build a one hot encoding table for all the operators in the network.
    # action_space: the parameters to be optimized by the AutoML service. should be a dictionary of parameter name and the available options to tweak said parameter. Each parameter should also hold a key for whether the tweak options are discrete or not.
    dataframe = {}
    # todo once dst_op map is automatically generated, change this
    if params['wlparams']['layers'] == 4:
        dst_ops = {f'seq.{i}': [f"seq.{i + 1}"] for i in range(3)}
        dst_ops['seq.3'] = []
    elif params['wlparams']['layers'] == 32:
        dst_ops = {f'seq.{i}': [f"seq.{i + 1}"] for i in range(31)}
        dst_ops['seq.31'] = []
    else:
        print('automatic wrappers for creating network graphs are not yet implemented')
        raise NotImplementedError
    dataframe['dst_op'] = dst_ops
    # todo once features encoding is automatically generated, change this
    # only one operator type contained in linear. So the one hot encoding has just one value
    if params['wlparams']['layers'] == 4:
        features = {f'seq.{i}': [1, 1024*1024, 1024*1024, 1024*1024] for i in range(1, 3)}
        features['seq.3'] = [1, 1024*10, 1024*1024, 10]
    elif params['wlparams']['layers'] == 32:
        features = {f'seq.{i}': [1, 1024*1024, 1024*1024, 1024*1024] for i in range(1, 31)}
        features['seq.31'] = [1, 1024*10, 1024*1024, 10]
    else:
        print('automatic wrappers for creating network graphs are not yet implemented')
        raise NotImplementedError
    features['seq.0'] = [1, 28*28*1024, 0, 1024*1024]
    dataframe['features'] = features
    options = [[True, False], 'discrete']
    action_space = {'action_type': {'gather_input': options, 'split_output': options, 'row_linear': options,
                                    'column_linear': options, 'gather_output': options, 'input_is_parallel': options,
                                    'data_parallel_input': options}}
    dataframe['action_space'] = action_space
    # # todo delete
    # if dist.get_rank() == 0:
    #     print(f'dataframe: \n{dataframe}')

    # Example:
    # dataframe = {"action_space":{"action_type":{"device_start_idx":[["\"0\"","\"1\""],"discrete"],"strategy":[["1,1,1,1","1,1,1,2","1,2,1,1","2,1,1,1"],"discrete"]}},"dst_op":{"Conv2D_100":["Pool2D_101"],"Conv2D_102":["Pool2D_103"],"Conv2D_104":["Conv2D_105"],"Conv2D_105":["Conv2D_106"],"Conv2D_106":["Pool2D_107"],"Dense_109":["Dense_110"],"Dense_110":["Dense_111"],"Dense_111":["Softmax_112"],"Flat_108":["Dense_109"],"Pool2D_101":["Conv2D_102"],"Pool2D_103":["Conv2D_104"],"Pool2D_107":["Flat_108"],"Softmax_112":[]},"features":{"Conv2D_100":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23296,0,51380224],"Conv2D_102":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,307392,11943936,35831808],"Conv2D_104":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,663936,8306688,16613376],"Conv2D_105":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,884992,16613376,11075584],"Conv2D_106":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,590080,11075584,11075584],"Dense_109":[0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,37752832,2359296,1048576],"Dense_110":[0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16781312,1048576,1048576],"Dense_111":[0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,40970,1048576,2560],"Flat_108":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,,0,0,0,0,2359296,2359296],"Pool2D_101":[0,0,0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51380224,11943936],"Pool2D_103":[0,0,0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35831808,8306688],"Pool2D_107":[0,0,0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11075584,2359296],"Softmax_112":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,,0,0,0,2560,2560]},"strategy":{}}
    return {'normalized_state_df': dataframe, 'rc': 0}


def evaluate(params,
             model, optimizer, loss, dp_x: torch.Tensor, targets: torch.Tensor, vanilla_latency, sim_iterations=10, version="v1"):
    """
    todo add documentation
    """
    # # todo delete
    # prYellow(f'rank {dist.get_rank()} entered evaluate with the following strategy: \n{params}')
    # todo delete this if the processes still have trouble syncing
    # wait for all processes to enter the evaluation phase
    dist.barrier()
    # todo delete these?
    # workload_fpath = params['wlparams']['workload_fpath']
    # options = params.get("options", None)
    # mapping = params['mapping']
    torch.cuda.empty_cache()
    # # todo delete
    # print(f'rank {dist.get_rank()} started simulation with the following strategy: \n{params}')
    next_strategy = {layer_name: LayerStrategy(**strat) for layer_name, strat in params.items()}
    # # todo delete
    # prYellow(f'rank {dist.get_rank()} created the following strategy: \n{next_strategy}')
    # Example of a "mapping" instance:
    # mapping = {"Conv2D_100":{"device_start_idx":0,"param1":1,"param2":1,"param3":1,"param4":1},"Pool2D_107":{"device_start_idx":0,"param1":1,"param2":1,"param3":1,"param4":4},"Softmax_112":{"device_start_idx":0,"param1":1,"param2":4,"param3":1,"param4":1}}

    # Evaluate
    # DOMAIN SPECIFIC CODEID
    # =========
    processing_time = time.time()
    num_errors = strategy_handler.count_errors(strategy=next_strategy)
    # todo change this to num_errors > 0
    if not strategy_handler.valid_parallel(next_strategy):
        # if the strategy is invalid, the latency multiplied by a factor of 2 * number of errors
        # todo delete
        if dist.get_rank() == 0:
            print(f'num errors: {num_errors}')
        next_latency = 3 * num_errors * vanilla_latency
    else:
        x = copy.deepcopy(dp_x)
        tgt = copy.deepcopy(targets)
        # # todo delete
        # if dist.get_rank() == 0:
        #     print(f'input shape: {dp_x.shape}. output shape: {targets.shape}')
        # wrap the model with the new strategy
        optimized_model, optimized_optimizer = copy.deepcopy(model), copy.deepcopy(optimizer)
        optimized_model = wrap_model(optimized_model, strategy=next_strategy)
        # check the time it takes to do iterations over the input data
        try:
            # todo delete
            print(f'rank {dist.get_rank()} started simulation with the following strategy: \n{next_strategy}')
            # # todo delete
            # if dist.get_rank() == 0:
            #     print(f'model: {optimized_model}\n model output shape: {optimized_model(x).shape}')
            next_latency = simulate(optimized_model, optimized_optimizer, loss, x, tgt, sim_iterations)
            # # todo delete
            # if dist.get_rank() == 0:
            #     print(f'rank {dist.get_rank()} finished simulation with the following latency: {next_latency} and the following strategy: {next_strategy}')
        except RuntimeError:
            # if the strategy is invalid, the latency increases by a factor of 3
            # todo find a smarter metric since this might overlap with valid strategies that are simply not efficient
            next_latency = 3 * vanilla_latency
            # todo delete
            if dist.get_rank() == 0:
                print(f'runtime error caught for strategy: \n{next_strategy}')
    # ips defined by batch / latency determined by simulation
    # # todo delete
    # if dist.get_rank() == 0:
    #     prGreen(
    #         f'rank {dist.get_rank()} finished simulation with the following latency: {next_latency}')
    # todo delete
    print(f'rank {dist.get_rank()} finished simulation with the following latency: \n{next_latency}, vanilla latency: {vanilla_latency}')
    # simulation yields: [latency] = [sec/inferences] = [sec/(batch*sim_iterations)] -> [ips] = [1/sec] = [1/(latency*sim_iterations*batch)]
    ips = targets.shape[0] / next_latency
    baselineIPS = targets.shape[0] / vanilla_latency
    meta_data = {"IPS": ips, "Latency": next_latency, "baseline_latency": vanilla_latency, "num_errors": num_errors}
                 # "legal_strategy": float(num_errors == 0)}
    rwd = 10 * (ips / baselineIPS)
    processing_time = time.time() - processing_time
    # =========
    rc_dict = {'rc': 0, 'done': 1, 'reward': rwd, 'processing_time': processing_time,
               'msg': 'Evaluate completed', 'meta_data': meta_data}
    # # todo delete
    # prYellow(f'rank {dist.get_rank()} finished the simulation with ips: \n{ips}')
    # # todo delete if it breaks anything
    # dist.barrier()
    return rc_dict


def find_optimal_strategy(model, optimizer, loss, sample_data, targets,
                          sim_iterations=10, imported_strat=None):
    # =============== strategy initialization
    assert dist.is_initialized()

    if imported_strat:
        best_strategy = strategy_handler.import_strategy(imported_strat)
    else:
        best_strategy = strategy_handler.create_vanilla_strategy(copy.deepcopy(model))
    model.train()
    # todo change this to support custom data parallel groups
    # get split batch and gather outputs
    dp_x = copy.deepcopy(sample_data)
    targets = _gather(targets, batch=True)
    # get base latency
    optimized_model, optimized_optimizer = wrap_model(copy.deepcopy(model), best_strategy), copy.deepcopy(optimizer)
    # warmup
    for i in range(sim_iterations):
        simulate(optimized_model, optimized_optimizer, loss, dp_x, targets, sim_iterations)
    torch.cuda.empty_cache()
    # baseline
    vanilla_latency = simulate(optimized_model, optimized_optimizer, loss, dp_x, targets, sim_iterations)
    torch.cuda.empty_cache()
    best_latency = vanilla_latency
    # todo delete
    if dist.get_rank() == 0:
        print(f'passed first simulation, vanilla latency: {best_latency}')
    # # todo delete
    # if dist.get_rank() == 0:
    #     print(f'urls: \n{urls}\nstrategy:\n{best_strategy}')
    # ================ main loop
    print("SPAAS worker started 2")
    if dist.get_rank() == 0:
        reset_old_jobs()
    killer = GracefulKiller()
    from tempfile import NamedTemporaryFile, TemporaryDirectory
    os.makedirs("/tmp_projects/projects/", exist_ok=True)
    t = TemporaryDirectory(dir='/tmp_projects/projects/', prefix='tmp')
    tmpfolder = t.name
    tmpfolder_graph = tmpfolder + "_graph"
    print("project dirs: ", tmpfolder, tmpfolder_graph)
    os.makedirs(tmpfolder_graph, exist_ok=True)
    try:
        # # todo find out what the host name and port should be
        # dist.TCPStore(is_master=True)
        # initialize a store to create a job request from server process and communicate strategies to all others
        strategy_store = dist.FileStore('tmp_filestore', dist.get_world_size())
        strategy_store.set_timeout(timedelta(seconds=(5 * jobtimeout)))
        while not killer.kill_now:
            if dist.get_rank() == 0:
                with open('/tmp/healthy', 'w') as f:
                    pass
                remaining_jobs = 0
                attempts = 5
                while attempts > 0:  # Try 5 times, because the service might not yet be available.
                    print("Get job from: " + urls['get_job'] +
                          f"?codeid={codeid}")
                    try:
                        req = requests.request("GET", urls['get_job'] +
                                               f"?codeid={codeid}", headers={}, data={})
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
                # # todo delete
                # print(f'request: \n{req}')
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
                        # cmdargs, error_code, error_msg = parse_cmdargs(job_definition['wlparams'])
                        # job_definition['wlparams']['params']['cmdargs'] = cmdargs
                        # if error_code:
                        #     print(f"Exception in evaluating job: {error_msg['msg']}")
                        #     requests.request("GET", urls['return_key'] + f"?key={last_active_key}", headers={}, data={})
                        #     continue

                        # ===========
                        # Process tickets from service:
                        # ===========
                        if GET_DATAFRAME_JOB in last_active_key:
                            # Get DataFrame
                            # todo delete
                            print(f'job definition: \n{job_definition}')
                            # todo decouple the mlp from the get dataframe job. delete the layers hardcoded definition once you have a way of extracting dataframe from nn.Module
                            job_definition['wlparams']['layers'] = len(model.seq)
                            results_json = get_data_frame(job_definition, tmpfolder)
                            # todo delete
                            print(f'results: \n{results_json}')
                            # Example of a results_json instance:
                            # results_json = {"normalized_state_df":{"action_space":{"action_type":{"device_start_idx":[["\"0\"","\"1\""],"discrete"],"strategy":[["1,1,1,1","1,1,1,2","1,2,1,1","2,1,1,1"],"discrete"]}},"dst_op":{"Conv2D_100":["Pool2D_101"],"Conv2D_102":["Pool2D_103"],"Conv2D_104":["Conv2D_105"],"Conv2D_105":["Conv2D_106"],"Conv2D_106":["Pool2D_107"],"Dense_109":["Dense_110"],"Dense_110":["Dense_111"],"Dense_111":["Softmax_112"],"Flat_108":["Dense_109"],"Pool2D_101":["Conv2D_102"],"Pool2D_103":["Conv2D_104"],"Pool2D_107":["Flat_108"],"Softmax_112":[]},"features":{"Conv2D_100":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23296,0,51380224],"Conv2D_102":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,307392,11943936,35831808],"Conv2D_104":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,663936,8306688,16613376],"Conv2D_105":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,884992,16613376,11075584],"Conv2D_106":[0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,590080,11075584,11075584],"Dense_109":[0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,37752832,2359296,1048576],"Dense_110":[0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16781312,1048576,1048576],"Dense_111":[0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,40970,1048576,2560],"Flat_108":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,,0,0,0,0,2359296,2359296],"Pool2D_101":[0,0,0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51380224,11943936],"Pool2D_103":[0,0,0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35831808,8306688],"Pool2D_107":[0,0,0,0,0,0,0,1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11075584,2359296],"Softmax_112":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,,0,0,0,2560,2560]},"strategy":{}}}
                        elif EVALUATE_MAPPING_JOB in last_active_key:
                            # Evaluate
                            # ========
                            # Example of a "job_definition"
                            # job_definition = {"issued_time":1625312390.1763651,"jobtype":"job","mapping":{"Conv2D_100":{"device_start_idx":0,"param1":1,"param2":1,"param3":1,"param4":1},"Pool2D_107":{"device_start_idx":0,"param1":1,"param2":1,"param3":1,"param4":4},"Softmax_112":{"device_start_idx":0,"param1":1,"param2":4,"param3":1,"param4":1}}}
                            # ========
                            # convert mapping into a list of booleans (a strategy)
                            mappings = {layer: {action: bool(val) for action, val in strategy.items()}
                                        for layer, strategy in job_definition['mapping'].items()}
                            mappings.pop('auxiliarly_params', 0)
                            # todo delete
                            print(f'mappings {mappings}')
                            strategy_store.set(STRATEGY_KEY, json.dumps(copy.deepcopy(mappings)))
                            # add a counter to make sure the strategy isnt reused
                            current_number = strategy_store.add(STRATEGY_SEEN_KEY, 0)
                            strategy_store.add(STRATEGY_SEEN_KEY, -current_number)
                            results_json = evaluate(params=mappings, model=model, optimizer=optimizer, loss=loss, dp_x=dp_x, targets=targets,
                                                    vanilla_latency=vanilla_latency, sim_iterations=sim_iterations, version="v1")
                            # todo uncomment this when you switch to tcp store
                            # strategy_store.delete_key('strategy')
                            next_latency = results_json['meta_data']['Latency']
                            if next_latency < best_latency:
                                best_strategy = copy.deepcopy(mappings)
                                best_latency = next_latency
                            # # todo delete
                            # print(f'results: \n{results_json}')
                        else:
                            print('invalid key received from service')
                            raise NotImplementedError

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

            else:
                # todo add try catch here
                # get mapping from dist.FileStore (distributed communication between processes)
                while strategy_store.add(STRATEGY_SEEN_KEY, 1) >= dist.get_world_size():
                    time.sleep(0.5)
                strategy = json.loads(strategy_store.get(STRATEGY_KEY))
                # todo make sure results are the same in all processes
                results_json = evaluate(strategy, model, optimizer, loss, dp_x, targets, vanilla_latency, sim_iterations, version="v1")

    except Exception as e:
        print("Exception: ", e)

        if last_active_key and dist.get_rank() == 0:
            print("Goodbye in Exception A", flush=True)
            requests.request("GET", urls['return_key'] + f"?key={last_active_key}", headers={}, data={})
        print("-" * 60)
        traceback.print_exc(file=sys.stdout)
        print("-" * 60)
    print("Goodbye EOP", flush=True)

    # todo make sure all processes get the same strategy at the end
    return best_strategy, best_latency
