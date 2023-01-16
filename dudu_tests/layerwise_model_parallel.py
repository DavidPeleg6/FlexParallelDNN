# coding=utf-8

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


from typing import Callable, Optional, Dict

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from fairscale.nn.model_parallel.initialize import get_model_parallel_rank, get_model_parallel_world_size
from fairscale.nn.model_parallel.mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
)
from fairscale.nn.model_parallel.utils import VocabUtility, divide_and_check_no_remainder
from fairscale.nn.model_parallel.layers import _initialize_affine_weight
from dudu_tests.strategy_struct import LayerStrategy, ParallelLayer
# from dudu_tests.layerwise_data_parallel import gather_from_data_parallel_region, scatter_to_data_parallel_region
import functools
import copy


def init_copy_method(tensor: torch.Tensor, other_tensor: torch.Tensor):
    with torch.no_grad():
        return tensor.copy_(other_tensor)


# todo find a way to add a conditional that makes sure the input is already data parallel


class ColumnParallelLinear(ParallelLayer):
    """Linear layer with column parallelism.
    todo fix documentation
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        layer: the linear layer that will be converted into column parallel linear
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        layer: torch.nn.Linear,
        layer_name: str,
        strategy: Dict[str, LayerStrategy] = None,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ) -> None:
        super().__init__(layer_name, strategy)
        # todo delete these
        # super(ColumnParallelLinear, self).__init__()
        # if strategy is not None and layer_name in strategy.keys():
        #     self.gather_output = strategy[layer_name].gather_output
        # else:
        #     self.gather_output = True
        # self.layer_name = layer_name
        # Keep input parameters
        self.in_features = layer.in_features
        self.out_features = layer.out_features

        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide_and_check_no_remainder(layer.out_features, world_size)

        # todo find a more elegant way to pass init_method if this works
        init_method = functools.partial(init_copy_method, other_tensor=copy.deepcopy(layer.weight.data))
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition, self.in_features))
        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.output_size_per_partition,
            0,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )
        if layer.bias is not None:
            bias_init_method = functools.partial(init_copy_method, other_tensor=copy.deepcopy(layer.bias.data.reshape
                                                                                              (layer.out_features, 1)))
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            _initialize_affine_weight(
                self.bias,
                self.out_features,
                1,
                self.output_size_per_partition,
                0,
                bias_init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
            )
            self.bias = Parameter(self.bias.reshape(self.output_size_per_partition))
        else:
            self.register_parameter("bias", None)

    # todo delete this
    # def get_strategy(self):
    #     return self.layer_name, LayerStrategy(column_linear=True, gather_output=self.gather_output)

    def extra_repr(self) -> str:
        super().extra_repr()
        repre = (
            f"{super().extra_repr()}, "
            f"column_parallel=true, "
            f"gather_output={self.strategy.gather_output}, "
        )
        return repre

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data.transpose(0, 1)).transpose_(0, 1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.strategy.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class RowParallelLinear(ParallelLayer):
    """Linear layer with row parallelism.
    TODO update documentation if everything works
    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        layer: the linear layer that will be converted into row parallel linear
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        layer: torch.nn.Linear,
        layer_name: str,
        strategy: Dict[str, LayerStrategy] = None,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ):
        super().__init__(layer_name, strategy)
        # todo delete if everythin works, uncomment otherwise
        # super(RowParallelLinear, self).__init__()
        # if strategy is not None and layer_name in strategy.keys():
        #     self.input_is_parallel = strategy[layer_name].input_is_parallel
        # else:
        #     self.input_is_parallel = False
        # self.layer_name = layer_name
        # Keep input parameters
        self.in_features = layer.in_features
        self.out_features = layer.out_features

        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide_and_check_no_remainder(layer.in_features, world_size)
        # todo find a more elegant way to pass init_method if this works
        init_method = functools.partial(init_copy_method, other_tensor=layer.weight.data)
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.input_size_per_partition,
            1,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )
        if layer.bias is not None:
            # todo delete this if everything works
            # self.bias = Parameter(torch.Tensor(self.out_features))
            # # Always initialize bias to zero.
            # with torch.no_grad():
            #     self.bias.zero_()
            self.bias = Parameter(copy.deepcopy(layer.bias.data))
        else:
            self.register_parameter("bias", None)

    # def get_strategy(self):
    #     return self.layer_name, LayerStrategy(row_linear=True, input_is_parallel=self.input_is_parallel)

    def extra_repr(self) -> str:
        super(RowParallelLinear, self).extra_repr()
        repre = (
            f"{super().extra_repr()}, "
            f"row_parallel=true, "
            f"input_is_parallel={self.strategy.input_is_parallel}, "
        )
        return repre

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        # Set up backprop all-reduce.
        if self.strategy.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output

