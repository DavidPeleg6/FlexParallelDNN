from typing import Dict
import json
import torch


class LayerStrategy:
    def __init__(self, gather_input=False, split_output=False, row_linear=False, column_linear=False,
                 gather_output=False, input_is_parallel=False, data_parallel_input=False):
        """
        A data structure class that represents the structure of each layer of the workload
        Arguments:
            gather_input: data parallel, whether to gather the minibatch from all machines/nodes
            split_output: data parallel, whether to split the batch into minibatches for all machines
            row_linear: model parallel, whether to split linear layers in the row dimension
            input_is_parallel: model parallel, for row linear, if the input from the previous layer is already split,
                               this parameter could save computation time if previous layer is also row parallel
            column_linear: model parallel, whether to split linear layers in the column dimension
            gather_output: model parallel, for column linear, whether to gather the output. could save computation time
                            if next layer is also column parallel
        """
        self.gather_input = gather_input
        self.split_output = split_output
        self.data_parallel_input = data_parallel_input
        self.row_linear = row_linear
        self.input_is_parallel = input_is_parallel
        self.column_linear = column_linear
        self.gather_output = gather_output
        # TODO check whether the dictionary gets changed with the variables in the layer strategy class
        self.strategy_dict = {
            'DataParallel': {
                'gather_input': self.gather_input,
                'split_output': self.split_output,
                'data_parallel_input': self.data_parallel_input},
            'ModelParallel': {
                'row_linear': self.row_linear, 'input_is_parallel': self.input_is_parallel,
                'column_linear': self.column_linear, 'gather_output': self.gather_output}
        }

    def __str__(self):
        return str(self.strategy_dict)

    # # todo uncomment if fails
    # def __get__(self, instance, owner):
    #     return self.strategy_dict

    def __eq__(self, other):
        return other.strategy_dict == self.strategy_dict

    def __repr__(self):
        return str(json.dumps(self.strategy_dict, indent=4))


def dict_strategy(strategy: Dict[str, LayerStrategy]) -> Dict[str, dict]:
    return {name: layer.strategy_dict for name, layer in strategy.items()}


class ParallelLayer(torch.nn.Module):
    def __init__(self, layer_name: str, strategy: Dict[str, LayerStrategy] = None):
        """
        todo add documentation
        """
        super().__init__()
        if strategy is not None and layer_name in strategy.keys():
            self.strategy = strategy[layer_name]
        else:
            # in case where all layers are column parallel linear when auto wrapping the model, gather output is set to
            # true in order to maintain correctness of computation
            # todo delete this and uncomment the gather_output version if fails
            self.strategy = LayerStrategy(data_parallel_input=True)
            # self.strategy = LayerStrategy(gather_output=True, data_parallel_input=True)
        self.layer_name = layer_name

    def get_strategy(self) -> (str, LayerStrategy):
        return self.layer_name, self.strategy

    def extra_repr(self) -> str:
        repre = f"custom_layer_name={self.layer_name}"
        return repre

