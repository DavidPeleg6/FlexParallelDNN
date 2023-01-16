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
from torch.utils.data.distributed import DistributedSampler
from fairscale.nn.model_parallel import initialize_model_parallel, destroy_model_parallel
from dudu_tests import Optimizer, strategy_handler
from fairscale.utils.testing import set_random_seed
# todo delete this once you have a better way
from dudu_tests.layerwise_data_parallel import _gather
import dudu_tests.autoML_optimizer as EgrlOptimizer
from enum import Enum


# todo after you combine the egrl and mcmc optimizer classes, move this to the combined optimizer
class OptimEnum(Enum):
    EGRL = 'EGRL'
    MCMC = 'MCMC'


class Net(nn.Module):
    def __init__(self, dropout=0.5, layers=20):
        super(Net, self).__init__()
        lin_layers = [nn.Linear(28 * 28, 1024)]
        for i in range(1, layers-1):
            lin_layers += [nn.Linear(1024, 1024)]
        lin_layers += [nn.Linear(1024, 10)]
        self.seq = nn.ModuleList(lin_layers)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        for layer in self.seq:
            x = layer(x)
            x = self.dropout(x)
        return x


def init_processes(train_args):
    dist.init_process_group(backend='nccl')
    backends = {"model_parallel_backend": "nccl", "pipeline_backend": "nccl", "ddp_backend": "nccl"}
    initialize_model_parallel(dist.get_world_size(), 1, **backends)
    local_rank = dist.get_rank()
    # todo change this so that randomness will be maintained across different processes
    set_random_seed(int(time.strftime("%M")))
    # cls.vanilla_strategy = strategy_handler.strategy_from_net(Optimizer.wrap_model(model=MnistMLP(), ))
    torch.cuda.set_device(local_rank)
    best_ips, latency = train_and_test(local_rank, train_args)
    if local_rank == 0:
        print(f'Best ips: {best_ips}, Latency: {latency}, Best strategy:\n{train_args.strategy} ')
    destroy_model_parallel()
    dist.destroy_process_group()


def train_and_test(rank, train_args):
    # =============================================================================
    # #Init
    # =============================================================================
    global_rank = dist.get_rank()
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

    dropout = 0.1
    n_workers = 1
    # optimizer
    optim_kwargs = {'lr': 2e-3, 'weight_decay': 1e-4, 'betas': (.9, .999)}
    # =============================================================================
    # Input
    # =============================================================================
    # 4) create dataset
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    # create data loader
    sampler = DistributedSampler(dataset, shuffle=False)
    # sampler = None
    kwargs = {'num_workers': n_workers, 'shuffle': False, 'drop_last': True, 'pin_memory': True,
              'batch_size': train_args.batch_size, 'sampler': sampler}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=train_args.batch_size)
    # =============================================================================
    # Find Optimal strategy
    # =============================================================================
    # todo you need to sync randomness across processes
    temp_model = Net(dropout, layers=train_args.layers)
    print(f'finding optimal strategy')
    loss_model = nn.CrossEntropyLoss()
    # for running without oss
    temp_optimizer = optim.Adam(temp_model.parameters(), **optim_kwargs)
    sample, targets = next(iter(data_loader))
    sample, targets = sample.to(rank), targets.to(rank)
    # todo uncomment this if the vanilla optimizer broke
    # targets = _gather(targets.to(rank), batch=True)
    if train_args.optimization_alg == OptimEnum.MCMC.value:
        train_args.strategy, best_latency = Optimizer.find_optimal_strategy(temp_model, temp_optimizer, loss_model, sample,
                                                             targets, train_args.budget, alpha=0.005,
                                                             sim_iterations=train_args.n_simulations,
                                                             imported_strat=train_args.import_strategy)
    elif train_args.optimization_alg == OptimEnum.EGRL.value:
        train_args.strategy, best_latency = EgrlOptimizer.find_optimal_strategy(temp_model, temp_optimizer, loss_model,
                                                                            sample, targets,
                                                                            sim_iterations=train_args.n_simulations,
                                                                            imported_strat=train_args.import_strategy)
    else:
        print('only mcmc and egrl are currently supported')
        raise NotImplementedError

    if rank == 0 and train_args.print_params:
        print(f'optimal latency: {best_latency}, optimal strategy:\n{train_args.strategy}')
    if train_args.export_strategy:
        strategy_handler.export_strategy(strategy=train_args.strategy, target_location=train_args.export_strategy)

    # =============================================================================
    # Train
    # =============================================================================
    # # todo delete this usage of streams if there are failures
    # s = torch.cuda.Stream()
    # with torch.cuda.stream(s):
    model = Optimizer.wrap_model(model=Net(dropout, layers=train_args.layers), strategy=train_args.strategy)
    loss_model = nn.CrossEntropyLoss()
    # for running without oss
    optimizer = optim.Adam(model.parameters(), **optim_kwargs)
    model.train()
    start = time.time()
    avg_ips = 0
    avg_latency = 0
    # profiler stuff
    if train_args.profile:
        profiler.start()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(train_args.epochs):
            train_loss = 0.0
            # TODO change this to be a bit more elegant instead of keeping two counters and breaking from a loop
            counter = 0
            max_iter = len(data_loader.dataset)
            # profiler stuff
            if train_args.profile:
                # iterations = (wait + warmup + active) * repeat
                max_iter = (2 + 2 + 6) * 3
            start0 = time.time()
            for data, target in data_loader:
                optimizer.zero_grad()
                data = data.to(rank)
                # target = target.to(rank)
                output = model(data)
                # todo delete and find a more elegant way to do this. maybe wrap the loss function with your own one?
                target = _gather(target.to(rank), batch=True)
                loss = loss_model(output, target)
                loss.backward()
                # extra step to make sure all gradients are in sync
                Optimizer.sync_gradients(model)
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                counter += 1
                if counter > max_iter:
                    break
                # profiler stuff
                if train_args.profile:
                    profiler.step()

            cur_time = time.time() - start0
            train_loss = train_loss / len(data_loader.dataset)
            latency = cur_time / counter
            avg_latency += cur_time
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
        avg_latency /= train_args.epochs
        print(f"RANK = {rank}, GPU AMOUNT = {dist.get_world_size()}, ELAPSED TIME = {tot_time}s, "
              f"BATCH = {train_args.batch_size}"
              f" AVG. THROUGHPUT = {avg_ips} samples/s, AVG. LATENCY = {avg_latency}")

    # =============================================================================
    # Test
    # =============================================================================
    # torch.save({"model": model.state_dict(), "strategy": train_args.strategy}, train_args.name)
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()    # prep model for *evaluation*

    for data, target in test_loader:
        data = data.to(rank)
        output = model(data)
        # todo delete and find a more elegant way to do this
        target = _gather(target.to(rank), batch=True)
        loss = loss_model(output, target)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        for i in range(data.shape[0]):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy (Overall): %.4f%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    return avg_ips, latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--n_simulations", type=int, default=8)
    parser.add_argument('--print_params', dest='print_params', default=False, action='store_true')
    parser.add_argument('--validate', dest='validate', default=False, action='store_true')
    parser.add_argument('--profile', dest='profile', default=False, action='store_true')
    parser.add_argument("--epochs", action="store", default=10, type=int)
    parser.add_argument("--budget", action="store", default=100, type=int)
    parser.add_argument("--import_strategy", action="store", default='', type=str)
    parser.add_argument("--export_strategy", action="store", default='', type=str)
    parser.add_argument("--name", action="store", default='no_name.pt', type=str)
    parser.add_argument("--optimization_alg", action="store", default=OptimEnum.MCMC, type=str)
    args = parser.parse_args()
    if args.import_strategy:
        args.imported_strat = args.import_strategy
        args.strategy = strategy_handler.import_strategy(args.imported_strat)

    init_processes(args)



