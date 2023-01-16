import torch
import json
from typing import Dict
from collections import OrderedDict
from dudu_tests import custom_wrap
import random
import functools
from dudu_tests.strategy_struct import LayerStrategy, dict_strategy, ParallelLayer
from dudu_tests.layerwise_data_parallel import DataParallelLayer
import copy


def import_strategy(strat_file):
    """
    A function to parse the strategy from a file
    Arguments:
        strat_file:
            the path to the strategy file to be analyzed a function to parse the strategy from file
    Return:
        strategy dictionary if the file exists, else None
    """
    try:
        with open(strat_file, 'r') as file:
            strategies = file.read()
        strategies = json.loads(strategies)
        # recreate dictionary with GatherSplit strategies
        strategies = {key: LayerStrategy(**(val['DataParallel']), **(val['ModelParallel']))
                      for key, val in strategies.items()}
        return strategies
    except FileNotFoundError:
        return None


def export_strategy(model: torch.nn.Module = None, strategy: Dict[str, LayerStrategy] = None,
                    target_location: str = '') -> dict:
    """
    Save a strategy for later use
    Arguments:
        strategy: a strategy dictionary to save
        target_location: the location to save the strategy in
    """
    if model is None and strategy is None:
        assert False
    elif model is not None:
        strategy = strategy_from_net(model)
    if target_location:
        with open(target_location, 'w') as file:
            file.write(json.dumps(dict_strategy(strategy), indent=4))
    return strategy


def create_vanilla_strategy(model: torch.nn.Module, wrapper_cls: ParallelLayer = DataParallelLayer,
                            min_num_params: int = 1, type_filter: torch.nn.Module = None) -> Dict[str, LayerStrategy]:
    """
    creates a vanilla strategy of a model given. if a target location is given, it will save the strategy to that
    location.
    currently chooses layers to implement strategy based on min number of parameters, but will later add option to
    accept any fucntion.
    Arguments:
        model: The model for strategy creation
        wrapper_cls: the class its wrapped in. For now its DataParallelLayer, but will be MixedParallelLayer
        min_num_params: the minimum amount of parameters a module of the network needs to have
        type_filter: when creating a strategy, you can choose to create it for specific type of layers only.
                    currently used for linear layers
    """
    data_params = dict(wrapper_cls=wrapper_cls)
    with custom_wrap.enable_wrap(**data_params):
        my_auto_wrap_policy = functools.partial(custom_wrap.default_auto_wrap_policy,
                                                min_num_params=int(min_num_params), type_filter=type_filter)
        temp_model = custom_wrap.auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy)
    strategy = OrderedDict(strategy_from_net(temp_model))
    strategy[list(strategy.keys())[-1]] = LayerStrategy(gather_input=True)
    return strategy


def strategy_from_net(model: torch.nn.Module):
    """
    Iterates over the given network and returns its strategy.
    Arguments:
        model: The model to create strategy for
    """
    strategy = {}
    for name, module in model.named_modules():
        try:
            strategy[name] = module.get_strategy()[1]
        except AttributeError:
            # strategy[name] = LayerStrategy()
            pass
    return strategy


def get_model_graph(model: torch.nn.Module):
    """
    Iterates over the given network and returns the base computation graph. used by EGRL
    Arguments:
        model: The model to create computation graph for
    """
    # todo test this function actually works. it doesnt. fix this (maybe using torch.fx?)
    dst_ops = {}
    for _, module in model.named_modules():
        try:
            dst_ops[module.layer_name] = [child.layer_name for _, child in module.named_children()]
        except AttributeError:
            pass
    return dst_ops


def randomize_strategy(strategy, random_layer_amount=False) -> Dict[str, LayerStrategy]:
    """
    choose a random layer from a given strategy, and randomize it
    TODO add an element of memory to all this. The memory element will be stored in optimizer. this entire function should probably be there
    main loop rather than the training loop
    Arguments:
        strategy: the strategy to randomize
        random_layer_amount: Whether to change a random amount of network layer's strategies. default is to change 1 layer
    Returns:
        strategy: the randomized strategy
    """
    original_strat = copy.deepcopy(strategy)
    # choose the amount of layers to change
    random_amount = random.randint(1, 2 * len(strategy.keys())) if random_layer_amount else 1
    random_layers = random.choices(list(strategy.keys()), k=random_amount)
    for layer in random_layers:
        # # todo delete this if norm layers dont cause a failure
        # if 'norm' in layer:
        #     continue
        # todo add a feature to choose randomize only within the correct search space.
        # todo create a function that returns the valid states space
        # choose between data and model parallel
        parallel_type = random.choice(list(strategy[layer].strategy_dict.keys()))
        # choose an action to change in that parallel type
        action = random.choice(list(strategy[layer].strategy_dict[parallel_type]))
        # choose to turn that action on/off and recreate the strategy of that layer accordingly
        new_actions = {action: random.choice([True, False])}
        strategy[layer] = LayerStrategy(**new_actions)

    # if parallelization strategy is invalid, switch back to old strategy
    return strategy if valid_parallel(strategy) else original_strat


"""
todo complete the bailout function
def bailout(strategy: Dict[str, LayerStrategy]) -> (int, Dict[str, LayerStrategy]):
    strategy = OrderedDict(strategy)
    # change into ordered dict to check that the first layer is not wrapped and the last layer gathers batch
    first_lay, last_lay = list(strategy.values())[0], list(strategy.values())[-1]
    # first_lay, last_lay = list(OrderedDict(strategy).values())[0], list(OrderedDict(strategy).values())[-1]
    if first_lay.gather_input or first_lay.split_output or first_lay.row_linear or first_lay.column_linear:
        return False
    if last_lay.split_output or last_lay.row_linear or last_lay.column_linear or last_lay.data_parallel_input:
        return False
    return check_valid_data_parallel(strategy) and check_valid_row_parallel(strategy) \
           and check_valid_column_parallel(strategy)
"""


def count_errors(strategy: Dict[str, LayerStrategy]) -> int:
    error_count = []
    strategy = OrderedDict(strategy)
    # change into ordered dict to check that the first layer is not wrapped and the last layer gathers batch
    first_lay, last_lay = list(strategy.values())[0], list(strategy.values())[-1]
    # first_lay, last_lay = list(OrderedDict(strategy).values())[0], list(OrderedDict(strategy).values())[-1]
    if first_lay.gather_input or first_lay.split_output or first_lay.row_linear or first_lay.column_linear:
        error_count.append(1)
    if last_lay.split_output or last_lay.row_linear or last_lay.column_linear or last_lay.data_parallel_input:
        error_count.append(1)
    check_valid_data_parallel(strategy, error_count)
    check_valid_row_parallel(strategy, error_count)
    check_valid_column_parallel(strategy, error_count)
    return sum(error_count)


def valid_parallel(strategy: Dict[str, LayerStrategy]) -> bool:
    """
    todo add documentation
    """
    strategy = OrderedDict(strategy)
    # change into ordered dict to check that the first layer is not wrapped and the last layer gathers batch
    first_lay, last_lay = list(strategy.values())[0], list(strategy.values())[-1]
    # first_lay, last_lay = list(OrderedDict(strategy).values())[0], list(OrderedDict(strategy).values())[-1]
    if first_lay.gather_input or first_lay.split_output or first_lay.row_linear or first_lay.column_linear:
        return False
    if last_lay.split_output or last_lay.row_linear or last_lay.column_linear or last_lay.data_parallel_input:
        return False
    return check_valid_data_parallel(strategy) and check_valid_row_parallel(strategy)\
           and check_valid_column_parallel(strategy)


def check_valid_data_parallel(strategy: Dict[str, LayerStrategy], error_counter=[]) -> bool:
    """
    Checks that the given strategy is valid by iterating over it and making sure all layers abide two rules:
    1. you cant gather a gathered input
    2. you cant split a split input
    * the underlying assumption is that the strategy given is in the same order as the network architecture (the flow
    through the strategy dict should be the same as the flow of the input through the network)
    Arguments:
        strategy: a mapping between each layer name and the action it performs on the input in terms of gather/split
    """
    # assuming the training starts from a data parallel state (meaning batch is already split)
    gathered = False
    valid = True
    for layer, action in strategy.items():
        # if some previous layer gathered and didnt split, you cant use gather
        if gathered and action.gather_input:
            valid = False
            error_counter.append(1)
            # todo uncomment if something broke
            # return False
        # if some previous layer split, you cant split again unless you gathered first
        if not gathered and not action.gather_input and action.split_output:
            valid = False
            error_counter.append(1)
            # todo uncomment if something broke
            # return False
        # if batch is split, all following layers until gather must be aware that their gradients should be synced
        if not (gathered or action.gather_input or action.split_output or action.data_parallel_input):
            valid = False
            error_counter.append(1)
            # todo uncomment if something broke
            # return False
        # if batch is gathered, dont sync gradients
        if gathered and action.data_parallel_input:
            valid = False
            error_counter.append(1)
            # todo uncomment if something broke
            # return False
        # update the input state
        if action.gather_input:
            gathered = True
        if action.split_output:
            gathered = False
    # in the final layer, batch must be gathered
    return gathered and valid


def check_valid_row_parallel(strategy: Dict[str, LayerStrategy], error_counter=[]) -> bool:
    """
    Checks that the given strategy is valid by iterating over it and making sure all layers abide two rules:
    1. you can only use row parallel with input_is_parallel = True if the previous layer was column parallel and
    gather_ouptut was set to False
    2. if the current layer is model parallel, data parallel must have been switched off
    * the underlying assumption is that the strategy given is in the same order as the network architecture (the flow
    through the strategy dict should be the same as the flow of the input through the network)
    Arguments:
        strategy: a mapping between each layer name and the action it performs on the input in terms of gather/split
    """
    # since training cant start with an input already split by the column dimension
    input_is_gathered = True
    # assuming the training starts from a data parallel state (meaning batch is already split)
    batch_gathered = False
    valid = True
    for layer, action in strategy.items():
        # if the current layer is model parallel and the previous layer was data parallel, return not valid
        if action.row_linear and not batch_gathered:
            valid = False
            error_counter.append(1)
            # todo uncomment if something broke
            # return False
        # if the previous layer wasnt column linear without gathered output, return false
        if input_is_gathered and action.row_linear and action.input_is_parallel:
            valid = False
            error_counter.append(1)
            # todo uncomment if something broke
            # return False
        if action.gather_input:
            batch_gathered = True
        if action.split_output:
            batch_gathered = False
        # update the input state
        if action.column_linear and not action.gather_output:
            input_is_gathered = False
            continue
        input_is_gathered = True
    return valid


def check_valid_column_parallel(strategy: Dict[str, LayerStrategy], error_counter=[]) -> bool:
    """
    Checks that the given strategy is valid by iterating over it and making sure all layers abide two rules:
    1. you can only use column parallel with gather_output = False if the next layer is a row parallel and
    input_is_parallel is set to True
    2. if the current layer is model parallel, data parallel must have been switched off
    * the underlying assumption is that the strategy given is in the same order as the network architecture (the flow
    through the strategy dict should be the same as the flow of the input through the network)
    Arguments:
        strategy: a mapping between each layer name and the action it performs on the input in terms of gather/split
    """
    # since training cant start with an input already split by the row dimension
    input_is_split = False
    # assuming the training starts from a data parallel state (meaning batch is already split)
    batch_gathered = False
    valid = True
    for layer, action in strategy.items():
        # if the current layer is model parallel and the previous layer was data parallel, return not valid
        if action.column_linear and not batch_gathered:
            valid = False
            error_counter.append(1)
            # todo uncomment if something broke
            # return False
        # if the previous layer was column linear that left the output split and this layer is not row linear with
        # input_is_parallel
        if input_is_split and not (action.row_linear and action.input_is_parallel):
            valid = False
            error_counter.append(1)
            # todo uncomment if something broke
            # return False
        # update the input state
        if action.column_linear and not action.gather_output:
            input_is_split = True
        if action.gather_input:
            batch_gathered = True
        if action.split_output:
            batch_gathered = False
    return valid

