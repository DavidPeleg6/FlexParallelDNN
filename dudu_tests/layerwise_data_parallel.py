# import libraries
import torch.distributed as dist
from typing import Any, Dict, Tuple
import torch
from fairscale.nn.model_parallel.initialize import get_data_parallel_group
from dudu_tests.strategy_struct import LayerStrategy, ParallelLayer


"""
note that the following functions were modified from the fairscale repo
https://github.com/facebookresearch/fairscale/blob/fecb665b812b6bfc38442e1fb1557e21508917f4/fairscale/nn/model_parallel/mappings.py#L47
https://github.com/facebookresearch/fairscale/blob/fecb665b812b6bfc38442e1fb1557e21508917f4/fairscale/nn/model_parallel/utils.py#L40
the recursive wrap of model layers is copied from fairscale autowrap.
"""
# TODO move all this stuff into mappings.py


class DataParallelLayer(ParallelLayer):
    """A wrapper for a pytorch layer that supports splitting the layer in the batch dimension
    This class contains two unique fields: gather_input and split_output.
    when we wish to reduce from data parallelism, well use gather_input. if the following layers also remove data
    parallelism, then split_output can also be set to false in order to avoid redundant computations

    Arguments:
        input_layer: an initialized torch.nn module layer
        layer_name: the name given to each layer. This field is used when converting flexflow strategies into fairscale
                    strategies
    """

    def __init__(
        self,
        input_layer: torch.nn.Module,
        layer_name: str,
        strategy: Dict[str, LayerStrategy] = None,
    ) -> None:
        super().__init__(layer_name, strategy)
        self.input_layer = input_layer
        self.weight = input_layer.weight
        try:
            self.bias = input_layer.bias
        except AttributeError:
            self.bias = None

    def get_layer(self):
        return self.input_layer

    def reduce_gradients(self):
        if self.strategy.data_parallel_input:
            torch.distributed.all_reduce(self.weight.grad)
            if self.bias is not None:
                torch.distributed.all_reduce(self.bias.grad)

    def forward(self, input_: torch.Tensor, *inputs, **kwargs) -> torch.Tensor:  # type: ignore
        # gather the tensor from data parallel - undo data parallelism
        input_parallel = gather_from_data_parallel_region(input_) if self.strategy.gather_input else input_
        # apply layer
        output_parallel = self.input_layer(input_parallel, *inputs, **kwargs)
        # split tensor back to data parallel - redo data parallelism
        output_parallel = scatter_to_data_parallel_region(output_parallel) if self.strategy.split_output \
            else output_parallel

        return output_parallel

    def extra_repr(self) -> str:
        repre = (
            f"{super().extra_repr()}, "
            f"gather_input={self.strategy.gather_input}, "
            f"split_output={self.strategy.split_output}, "
        )
        return repre


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False, batch: bool = False
) -> Tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
        TODO combine this function with the custom wrap
        batch: If True, split the tensor along its batch dimension (for data parallel)
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1 if not batch else 0
    last_dim_size = divide_and_check_no_remainder(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def _reduce(ctx: Any, input_: torch.Tensor, batch=False) -> torch.Tensor:
    """All-reduce the the input tensor across model parallel group.
    """
    # todo uncomment once you have per-layer hybrid parallel (mix of model and data)
    # group = get_data_parallel_group()
    group = None

    if ctx:
        ctx.mark_dirty(input_)

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=group)
    # input_ /= torch.distributed.get_world_size(group=group)

    return input_


def _split(input_: torch.Tensor, batch=False) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    # todo uncomment once you have per-layer hybrid parallel (mix of model and data)
    # group = get_data_parallel_group()
    group = None

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Split along last dimension.
    world_size = torch.distributed.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size, batch=batch)

    # Note: torch.split does not create contiguous tensors by default.
    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor, batch=False) -> torch.Tensor:
    """Gather tensors and concatenate along the last dimension."""
    # todo uncomment once you have per-layer hybrid parallel (mix of model and data)
    # group = get_data_parallel_group()
    group = None
    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1 if not batch else 0
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


class _CopyToDataParallelRegion(torch.autograd.Function):
    """Pass the input to the data parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        # # todo delete this if fail
        # _reduce(ctx, grad_output, batch=True)
        # grad_output /= torch.distributed.get_world_size()
        # return grad_output
        return _reduce(None, grad_output, batch=True)


class _ReduceFromDataParallelRegion(torch.autograd.Function):
    """All-redcue the input from the data parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _reduce(ctx, input_, batch=True)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output


class _ScatterToDataParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _split(input_, batch=True)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        # todo add prints here to see exactly what happens
        return _gather(grad_output, batch=True)
        # print(f'scatter grad, before reduce:\n {grad_output}')
        # grad_output = _reduce(None, _gather(grad_output, batch=True), batch=True)
        # # todo delete
        # # grad_output = _reduce(None, grad_output, batch=True)
        # # todo make variable world size based on group here
        # # group = get_data_parallel_group()
        # group = None
        # grad_output.data /= dist.get_world_size(group=group)
        # # todo delete this
        # # print(f'scatter grad, after reduce:\n {grad_output}')
        # return grad_output


class _GatherFromDataParallelRegion(torch.autograd.Function):
    """Gather the input from data parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _gather(input_, batch=True)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        # splitted = grad_output
        splitted = _split(grad_output, batch=True)
        return splitted


def copy_to_data_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToDataParallelRegion.apply(input_)


def reduce_from_data_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromDataParallelRegion.apply(input_)


def scatter_to_data_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToDataParallelRegion.apply(input_)


def gather_from_data_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromDataParallelRegion.apply(input_)


