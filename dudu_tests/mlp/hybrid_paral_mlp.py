"""
A distributed mnist mlp using fsdp and fairscale.
This serves as a playground for experimentation with model and data parallel strategies similar to flexflow.
For a fully documented use case of fairscale, refer to multi_node_train.py
"""

# import libraries
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import os
import time
import argparse
import torch.distributed as dist
import functools
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, auto_wrap, default_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
from dudu_tests.layerwise_data_parallel import DataParallelLayer
from dudu_tests.layerwise_model_parallel import ColumnParallelLinear, RowParallelLinear
from dudu_tests import custom_wrap
from dudu_tests import strategy_handler
from fairscale.nn.model_parallel import initialize_model_parallel
import random
import json


class Net(nn.Module):
    def __init__(self, layers=8, dropout=0.2):
        super(Net, self).__init__()
        self.seq = torch.nn.Sequential()
        self.seq.add_module('fc0', nn.Linear(28 * 28, 512))
        for i in range(1, layers-1):
            self.seq.add_module(f'fc{i}', nn.Linear(512, 512))
        self.seq.add_module(f'fc{layers-1}', nn.Linear(512, 10))

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.seq(x)


def init_processes(train_args):
    local_rank = int(os.environ['LOCAL_RANK'])
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    # currently its assumed that it will use the default initialization method to sync all workers across nodes
    dist.init_process_group(backend='nccl')
    # initialize model parallel process: split layer across 1 machines and pipeline length is 1
    # data parallel size = world_size / (model * pipeline parallel)
    # TODO ask for support for mpi in the pod and run it this way (which will probably enable multi node run)
    # backends = {"model_parallel_backend": "nccl", "pipeline_backend": "mpi", "ddp_backend": "nccl"}
    backends = {"model_parallel_backend": "nccl", "pipeline_backend": "nccl", "ddp_backend": "nccl"}
    initialize_model_parallel(1, 1, **backends)

    # TODO seems put this code segment in a more elegant place (and make generate a strategy for random workload)
    if train_args.import_strategy:
        train_args.strategy = strategy_handler.import_strategy(train_args.import_strategy)
    # if no valid strategy is given, create a new one using the wrap feature
    else:
        train_args.strategy = strategy_handler.create_vanilla_strategy(model=Net(), wrapper_cls=DataParallelLayer,)
        strategy_handler.export_strategy(train_args.strategy, train_args.export_strategy)
    # warmup calibration run
    # simulator phase means whether to train to completion
    train_args.simulator_phase = False
    # train_args.simulator_phase = True
    best_ips = baseline = train_and_test(local_rank, train_args)
    if dist.get_rank() == 0 and train_args.print_params:
        # print(json.dumps(strategy_handler.dict_strategy(train_args.strategy)))
        print(f'vanilla strategy, cur_ips: {best_ips}')
    # seed is equal for all processes to ensure random strategies are constant between machines
    # train_args.simulator_phase = True
    train_args.simulator_phase = True
    random.seed(1)
    start = time.time()
    best_strategy = train_args.strategy.copy()
    seen_strategies = [best_strategy]
    for i in range(train_args.budget):
        iter_counter = 0
        # TODO delete this and return the memory feature
        while train_args.strategy in seen_strategies and iter_counter < 100:
        # while train_args.strategy in seen_strategies and iter_counter < 100:
            # randomize the strategy
            train_args.strategy = strategy_handler.randomize_strategy(best_strategy.copy(), random_layer_amount=True)
            iter_counter += 1
        seen_strategies.append(train_args.strategy.copy())

        # test the strategy by checking ips in training
        avg_ips = train_and_test(local_rank, train_args)
        # if its not good change the layer strategy back
        if avg_ips > best_ips:
            best_strategy = train_args.strategy.copy()
            best_ips = avg_ips
        if dist.get_rank() == 0 and train_args.print_params:
            print(f'iteration: {i}, cur_ips: {avg_ips}')
    simtime = (time.time() - start) / train_args.budget if train_args.budget else 0
    if dist.get_rank() == 0:
        print(f'Rank: {dist.get_rank()}, '
              f'Best ips: {best_ips}, '
              f'Baseline ips: {baseline}, '
              f'single simulation time: {simtime}\n'
              f'Best strategy: {json.dumps(strategy_handler.dict_strategy(best_strategy), indent=4)}')
        strategy_handler.export_strategy(best_strategy, train_args.export_strategy)
    dist.destroy_process_group()


def train_and_test(rank, train_args):
    # =============================================================================
    # #Init
    # =============================================================================
    global_rank = dist.get_rank()
    # print('initializing..')
    print(f"[{os.getpid()}] rank = {global_rank}, " + f"world_size = {dist.get_world_size()}")
    torch.cuda.set_device(rank)
    # profiler stuff
    if train_args.profile:
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=2,
                active=6,
                repeat=3),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/mnist_mlp{global_rank}'),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/mnist_mlp',
                                                                    worker_name=f'time:{time.localtime().tm_min}:{time.localtime().tm_sec},rank:{global_rank}'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()

    # =============================================================================
    # Input
    # =============================================================================
    # 4) create dataset
    # TODO change to 0 if you want the errors to stop
    # n_workers = 0
    n_workers = 1
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    # create data loader
    sampler = DistributedSampler(dataset, shuffle=False)
    kwargs = {'num_workers': n_workers, 'shuffle': False, 'drop_last': True, 'pin_memory': True,
              'batch_size': train_args.batch_size, 'sampler': sampler}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    # optimizer
    optim_kwargs = {'lr': 2e-3, 'weight_decay': 1e-4, 'betas': (.9, .999)}
    dropout = 0.1

    # =============================================================================
    # Model
    # =============================================================================
    model = Net(dropout=dropout)

    # wrap the layers of the model with data parallel layers that will enable custom data parallelism
    data_params = dict(wrapper_cls=DataParallelLayer, strategy=train_args.strategy)
    with custom_wrap.enable_wrap(**data_params):
        my_auto_wrap_policy = functools.partial(custom_wrap.data_parallel_wrap_policy, strategy=train_args.strategy,
                                                min_num_params=int(1))
        model = custom_wrap.auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy)

    # # wrap the layers of the model with model parallel layers that will enable custom model parallelism
    data_params = dict(wrapper_cls=ColumnParallelLinear, strategy=train_args.strategy)
    with custom_wrap.enable_wrap(**data_params):
        my_auto_wrap_policy = functools.partial(custom_wrap.column_parallel_wrap_policy, strategy=train_args.strategy,
                                                min_num_params=int(1))
        model = custom_wrap.auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy)

    # # wrap the layers of the model with model parallel layers that will enable custom model parallelism
    data_params = dict(wrapper_cls=RowParallelLinear, strategy=train_args.strategy)
    with custom_wrap.enable_wrap(**data_params):
        my_auto_wrap_policy = functools.partial(custom_wrap.row_parallel_wrap_policy, strategy=train_args.strategy,
                                                min_num_params=int(1))
        model = custom_wrap.auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy)

    # wrap the layers of the model with fsdp layers that will enable sharding the model weights
    fsdp_params = dict(wrapper_cls=FSDP, mixed_precision=False, flatten_parameters=True)
    with enable_wrap(**fsdp_params):
        # count parameters on the master node (verbose mode prints)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if rank == 0:
            if train_args.print_params:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f'name: {name}, size: {param.numel()}')
            print(f"total trainable parameter amount: {param_count}")
        # wrap the model
        # my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=int(1e9))
        my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=int(param_count))
        model = auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy, verbose=train_args.print_params)
        # verbose mode prints
    if rank == 0 and train_args.print_params:
        print(model)
    model = model.to(global_rank)

    # =============================================================================
    # Optimizer
    # =============================================================================
    print(f'initializing optimizer and loss  in global rank {global_rank}')
    loss_model = nn.CrossEntropyLoss()
    # for running without oss
    optimizer = optim.Adam(model.parameters(), **optim_kwargs)
    # =============================================================================
    # Train
    # =============================================================================
    model.train()
    start = time.time()
    avg_ips = 0
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(train_args.epochs):
            train_loss = 0.0
            # TODO change this to be a bit more elegant instead of keeping two counters and breaking from a loop
            counter = 0
            max_iter = 40 if train_args.simulator_phase else len(data_loader.dataset)
            # profiler stuff
            if train_args.profile:
                # iterations = (wait + warmup + active) * repeat
                max_iter = (2 + 2 + 6) * 3
            start0 = time.time()
            for data, target in data_loader:
                optimizer.zero_grad()

                data = data.to(rank)
                target = target.to(rank)

                output = model(data)
                loss = loss_model(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                counter += 1
                if counter > max_iter:
                    break
                # profiler stuff
                if train_args.profile:
                    profiler.step()

            train_loss = train_loss / len(data_loader.dataset)
            cur_time = time.time() - start0
            # ips = len(data_loader.dataset) / cur_time
            ips = counter * train_args.batch_size * dist.get_world_size() / cur_time
            # calculate the ips without the first iteration which contains the overhead of loading memory to the gpus
            avg_ips += ips
            if rank == 0 and train_args.print_params:
                print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
                print(f"RANK = {rank}, GPU AMOUNT = {dist.get_world_size()}, ELAPSED TIME = {cur_time}s,"
                      f" THROUGHPUT = {ips} samples/s")

        # profiler stuff
        if train_args.profile:
            profiler.stop()
        tot_time = time.time() - start
        # ips = ((train_args.epochs-1) * train_args.batch_size * dist.get_world_size()) / tot_time
        # avg_ips /= (train_args.epochs - 1)
        avg_ips /= train_args.epochs
        print(f"RANK = {rank}, GPU AMOUNT = {dist.get_world_size()}, ELAPSED TIME = {tot_time}s,"
              f" AVG. THROUGHPUT = {avg_ips} samples/s")
        torch.save({"model": model.state_dict(), "strategy": train_args.strategy}, train_args.name)

        if not train_args.simulator_phase:
            validate(rank, model, train_args.batch_size)

    return avg_ips


def validate(rank, model: nn.Module, batch_size):
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()  # prep model for *evaluation*
    loss_model = nn.CrossEntropyLoss()

    for data, target in test_loader:
        target = target.to(rank)
        data = data.to(rank)
        output = model(data)
        loss = loss_model(output, target)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        for i in range(data.shape[0]):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy (Overall): %.4f%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--print_params', dest='print_params', default=False, action='store_true')
    parser.add_argument('--validate', dest='validate', default=False, action='store_true')
    parser.add_argument('--profile', dest='profile', default=False, action='store_true')
    parser.add_argument("--epochs", action="store", default=10, type=int)
    parser.add_argument("--budget", action="store", default=100, type=int)
    parser.add_argument("--import_strategy", action="store", default='', type=str)
    parser.add_argument("--export_strategy", action="store", default='strategies/exported_strategy.json', type=str)
    parser.add_argument("--name", action="store", default='no_name.pt', type=str)
    args = parser.parse_args()

    init_processes(args)



