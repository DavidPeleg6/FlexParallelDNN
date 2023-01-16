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
# from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
# from fairscale.nn.wrap import enable_wrap, auto_wrap, default_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
# from dudu_tests.layerwise_data_parallel import DataParallelLayer
# from dudu_tests.layerwise_model_parallel import ColumnParallelLinear, RowParallelLinear
# from dudu_tests import custom_wrap
# from dudu_tests import strategy_handler
# from fairscale.nn.model_parallel import initialize_model_parallel
import random
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from ptsampler import Sampler


class Net(nn.Module):
    def __init__(self, dropout=0.5, layers=4):
        super(Net, self).__init__()
        lin_layers = [nn.Linear(28 * 28, 512)]
        for i in range(1, layers-1):
            lin_layers += [nn.Linear(512, 512)]
        lin_layers += [nn.Linear(512, 10)]
        self.seq = nn.ModuleList(lin_layers)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        for layer in self.seq:
            x = layer(x)
            x = self.dropout(x)
        return x


def init_processes(train_args):
    local_rank = int(os.environ['LOCAL_RANK'])
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    random.seed(1)
    # currently its assumed that it will use the default initialization method to sync all workers across nodes
    dist.init_process_group(backend='nccl')
    # warmup calibration run
    # # simulator phase means whether to train to completion
    # train_args.simulator_phase = False
    # warmup_ips = train_and_test(local_rank, train_args)
    # if dist.get_rank() == 0 and train_args.print_params:
    #     # print(json.dumps(strategy_handler.dict_strategy(train_args.strategy)))
    #     print(f'vanilla strategy, cur_ips: {warmup_ips}')
    # Actual run
    train_args.simulator_phase = False
    avg_ips = train_and_test(local_rank, train_args)
    if dist.get_rank() == 0:
        print(f'Rank: {dist.get_rank()}, '
              f'ips: {avg_ips}, ')
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

    # =============================================================================
    # Model
    # =============================================================================
    dropout = 0.1
    model = DDP(Net(dropout=dropout, layers=train_args.layers).to(rank))

    # =============================================================================
    # Optimizer
    # =============================================================================
    print(f'initializing optimizer and loss  in global rank {global_rank}')
    loss_model = nn.CrossEntropyLoss()
    # optimizer
    optim_kwargs = {'lr': 2e-3, 'weight_decay': 1e-4, 'betas': (.9, .999)}
    # optim_kwargs = {'lr': 2e-3, 'weight_decay': 1e-4}
    # for running without oss
    optimizer = optim.Adam(model.parameters(), **optim_kwargs)
    # count parameters on the master node (verbose mode prints)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        if train_args.print_params:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f'name: {name}, size: {param.numel()}')
        print(f"total trainable parameter amount: {param_count}")
        print(f'model: \n{model}')
    # =============================================================================
    # Train
    # =============================================================================
    model.train()
    avg_latency = 0
    start = time.time()
    avg_ips = 0
    sam = Sampler()
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
            for batch_idx, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()

                data = data.to(rank)
                target = target.to(rank)

                if batch_idx == 3 and epoch == 0:
                    sam.start()
                output = model(data)
                loss = loss_model(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx == 3 and epoch == 0:
                    sam.stop()
                train_loss += loss.item() * data.size(0)
                counter += 1
                if counter > max_iter:
                    break
                # profiler stuff
                if train_args.profile:
                    profiler.step()
            print(f'rank {rank} epoch {epoch} completed')

            train_loss = train_loss / len(data_loader.dataset)
            cur_time = time.time() - start0
            avg_latency += cur_time
            # ips = len(data_loader.dataset) / cur_time
            ips = counter * train_args.batch_size * dist.get_world_size() / cur_time
            # calculate the ips without the first iteration which contains the overhead of loading memory to the gpus
            avg_ips += ips
            if rank == 0 and train_args.print_params:
                print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
                print(f"RANK = {rank}, GPU AMOUNT = {dist.get_world_size()}, ELAPSED TIME = {cur_time}s,"
                      f" THROUGHPUT = {ips} samples/s")

        n = sam.graph()
        n.save_as_json(f"network_gpu{rank}.json")
        # profiler stuff
        if train_args.profile:
            profiler.stop()
        tot_time = time.time() - start
        # ips = ((train_args.epochs-1) * train_args.batch_size * dist.get_world_size()) / tot_time
        # avg_ips /= (train_args.epochs - 1)
        avg_ips /= train_args.epochs
        avg_latency /= train_args.epochs
        print(f"RANK = {rank}, GPU AMOUNT = {dist.get_world_size()}, ELAPSED TIME = {tot_time}s, "
              f"BATCH = {train_args.batch_size},"
              f" AVG. THROUGHPUT = {avg_ips} samples/s,  AVG. LATENCY = {avg_latency}")
        # torch.save({"model": model.state_dict()}, train_args.name)

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
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument('--print_params', dest='print_params', default=False, action='store_true')
    parser.add_argument('--validate', dest='validate', default=False, action='store_true')
    parser.add_argument('--profile', dest='profile', default=False, action='store_true')
    parser.add_argument("--epochs", action="store", default=10, type=int)
    parser.add_argument("--budget", action="store", default=10, type=int)
    parser.add_argument("--import_strategy", action="store", default='', type=str)
    parser.add_argument("--export_strategy", action="store", default='strategies/exported_strategy.json', type=str)
    parser.add_argument("--name", action="store", default='no_name.pt', type=str)
    args = parser.parse_args()

    init_processes(args)


