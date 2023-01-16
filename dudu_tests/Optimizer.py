from torch import nn
import torch
from dudu_tests.layerwise_data_parallel import DataParallelLayer
from dudu_tests.layerwise_model_parallel import ColumnParallelLinear, RowParallelLinear
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from dudu_tests import custom_wrap
from dudu_tests import strategy_handler
import functools
import copy
import time
import numpy.random as rand
import numpy as np
from dudu_tests.layerwise_data_parallel import _gather


MAX_RETRIES = 1000


def wrap_model(model: nn.Module, strategy: dict = None, print_params: bool = False, data_paral_wrapper: str = 'DDP'):
    """
    todo add documentation here
    """
    if strategy is None:
        strategy_handler.create_vanilla_strategy(model=model, )
        return model
    data_params = dict(wrapper_cls=DataParallelLayer, strategy=strategy)
    with custom_wrap.enable_wrap(**data_params):
        my_auto_wrap_policy = functools.partial(custom_wrap.data_parallel_wrap_policy, strategy=strategy,
                                                min_num_params=int(1))
        model = custom_wrap.auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy)

    data_params = dict(wrapper_cls=ColumnParallelLinear, strategy=strategy)
    with custom_wrap.enable_wrap(**data_params):
        my_auto_wrap_policy = functools.partial(custom_wrap.column_parallel_wrap_policy, strategy=strategy,
                                                min_num_params=int(1))
        model = custom_wrap.auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy)

    data_params = dict(wrapper_cls=RowParallelLinear, strategy=strategy)
    with custom_wrap.enable_wrap(**data_params):
        my_auto_wrap_policy = functools.partial(custom_wrap.row_parallel_wrap_policy, strategy=strategy,
                                                min_num_params=int(1))
        model = custom_wrap.auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy)

    # count parameters on the master node
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    group = None
    model = model.to(dist.get_rank(group=group))
    # # todo delete this maybe
    # # data_params = dict(wrapper_cls=FSDP)
    # # with enable_wrap(**data_params):
    # #     my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=int(param_count))
    # #     model = auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy)
    # model.seq[0] = FSDP(model.seq[0])
    # if dist.get_rank() == 0:
    #     print(model)

    # todo maybe add group here
    # # todo this is the problem!!!!
    # if data_paral_wrapper == 'DDP':
    #     model = model.to(dist.get_rank(group=group))
    #     model = DDP(model)
    # elif data_paral_wrapper == 'FSDP':
    #     model = FSDP(model, verbose=print_params)
    #     model = model.to(dist.get_rank(group=group))

    if dist.get_rank() == 0 and print_params:
        print(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'name: {name}, size: {param.numel()}')
        print(f"total trainable parameter amount: {param_count}")

    return model


def sync_gradients(model: torch.nn.Module):
    """
    updates the appropriate gradients according to the strategy
    """
    # # todo delete if it doesnt fix anything or causes too much of an overhead
    # torch.cuda.synchronize()
    for layer in model.modules():
        try:
            layer.reduce_gradients()
        except AttributeError:
            pass


def simulate(model, optimizer, loss, x, targets, max_iterations):
    start = time.time()
    for j in range(max_iterations):
        optimizer.zero_grad()
        y = model(x)
        loss(y, targets).backward()
        sync_gradients(model)
        optimizer.step()
    return (time.time() - start) / max_iterations


def find_optimal_strategy(model, optimizer, loss, sample_data, targets, budget: int = 100, alpha=0.05,
                          sim_iterations=10, imported_strat=None):
    """
    todo add documentation, and find better way to extrapolate sample data from the model
    mcmc as written in flexflow (might not be an actual mcmc)
    todo change to metropolis hastings sampling method
    """
    # # todo delete this if theres some failure
    # s = torch.cuda.Stream()
    # with torch.cuda.stream(s):
    # get starting strategy
    if imported_strat:
        best_strategy = strategy_handler.import_strategy(imported_strat)
    else:
        best_strategy = strategy_handler.create_vanilla_strategy(copy.deepcopy(model))
    seen_strategies = [best_strategy]
    model.train()
    # todo change this to support custom data parallel groups
    # split batch
    dp_x = copy.deepcopy(sample_data)
    # todo maybe find a more elegant way to do this. (maybe using a custom loss function that gathers before calc?)
    # todo test that this change works
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
    cur_strategy, cur_latency = copy.deepcopy(best_strategy), copy.deepcopy(best_latency)
    for i in range(budget):
        if dist.get_rank() == 0 and i % 10 == 0:
            print('iteration: {}, current latency: {:.4f}, best latency: {:.4f} baseline latency: {:.4f}'.format(i, cur_latency, best_latency, vanilla_latency))
        # randomize a new strategy, if maximum no new strategy is found, loop until maximum
        next_strategy = strategy_handler.randomize_strategy(copy.deepcopy(cur_strategy), random_layer_amount=True)
        for j in range(MAX_RETRIES):
            if next_strategy not in seen_strategies:
                seen_strategies.append(next_strategy)
                break
            next_strategy = strategy_handler.randomize_strategy(copy.deepcopy(cur_strategy), random_layer_amount=True)
            if j == MAX_RETRIES - 1 and dist.get_rank() == 0:
                print('no new strategies could be found')
        # wrap the model with the new strategy
        optimized_model, optimized_optimizer = copy.deepcopy(model), copy.deepcopy(optimizer)
        optimized_model = wrap_model(optimized_model, strategy=next_strategy)
        # check the time it takes to do iterations over the input data
        try:
            next_latency = simulate(optimized_model, optimized_optimizer, loss, dp_x, targets, sim_iterations)
        except RuntimeError:
            # todo delete
            if dist.get_rank() == 0:
                print(f'runtime error caught for strategy in iteration{i}: \n{next_strategy}')
            continue
        if next_latency < best_latency:
            best_strategy = copy.deepcopy(next_strategy)
            best_latency = next_latency
        # mcmc step
        diff = next_latency - cur_latency
        step_prob = rand.uniform(0, 1)
        if next_latency < cur_latency:
            cur_strategy = copy.deepcopy(next_strategy)
            cur_latency = next_latency
        elif step_prob < np.e ** (-alpha * diff):
            cur_strategy = copy.deepcopy(next_strategy)
            cur_latency = next_latency
        torch.cuda.empty_cache()

    return best_strategy, best_latency

