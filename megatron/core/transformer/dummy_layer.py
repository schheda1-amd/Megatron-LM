import torch
import torch.nn as nn
from functools import partial, lru_cache
from typing import Optional, Tuple
from megatron.core import parallel_state
from megatron.core.utils import is_torch_min_version

if is_torch_min_version("2.4.0a0"):
    custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
    custom_bwd = partial(torch.amp.custom_bwd, device_type="cuda")
else:
    custom_fwd = torch.cuda.amp.custom_fwd
    custom_bwd = torch.cuda.amp.custom_bwd

dist_group_type = torch.distributed.ProcessGroup


@lru_cache
def get_distributed_world_size(group: Optional[dist_group_type] = None) -> int:
    """Return world size for the distributed group."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)

def gather_along_first_dim(
    input_: torch.Tensor,
    process_group: dist_group_type,
    async_op: bool = False,
) -> tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """All-gather tensors and concatenate along first dimension."""

    # Return immediately if no communication is required
    world_size = get_distributed_world_size(process_group)
    if world_size == 1:
        return input_, None

    # Output tensor dims
    print('whats the type? ', type(input_))
    out_shape = list(input_.size())
    out_shape[0] *= world_size

    # Communication for plain PyTorch tensors
    out = torch.empty(
        out_shape,
        dtype=input_.dtype,
        device=input_.device,
        memory_format=torch.contiguous_format,
    )
    handle = torch.distributed.all_gather_into_tensor(
        out,
        input_.contiguous(),
        group=process_group,
        async_op=async_op,
    )
    return out, handle


def _scatter_along_first_dim(
    input_: torch.Tensor, tp_group: dist_group_type, async_op: bool = False
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_distributed_world_size(tp_group)
    # Bypass the function if we are using only 1 GPU or 1 partition.
    if world_size == 1:
        return input_, None

    dim_size = list(input_.size())
    assert (
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by xcd model parallel size"

    #dim_size[0] = dim_size[0] // world_size
    if torch.distributed.get_rank(tp_group) == 0:
        scatter_list = [input_.chunk(world_size, dim=0)[i] for i in range(world_size)]
    else:
        scatter_list = None

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    handle = torch.distributed.scatter(
        output,
        scatter_list=scatter_list,
        group=tp_group,
        async_op=async_op,
    )
    return output, handle


class _DummyInpManipulation(torch.autograd.Function):
    """Dummy operator that will scatter the inputs in forward and gather in backward.
    This should be addressed by a reduce scatter in the RowParallelLinear's forward pass.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input_,):
        """Forward with frozen weight."""
        xcd_group = parallel_state.get_xcd_intra_gpu_parallel_group()
        xcd_group_size = parallel_state.get_xcd_intra_gpu_parallel_world_size()
        xcd_rank = parallel_state.get_xcd_intra_gpu_parallel_rank()
        
        input_chunks = torch.chunk(input_, xcd_group_size, dim=0)
        output = input_chunks[xcd_rank].contiguous()
        ctx.xcd_group = xcd_group
        ctx.xcd_group_size = xcd_group_size
        ctx.input_shape = input_.shape

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Backward to reconstruct the full grad."""
        xcd_group = ctx.xcd_group
        xcd_group_size = ctx.xcd_group_size
        input_shape = ctx.input_shape
        
        grad_chunks = [torch.zeros_like(grad_output) for _ in range(xcd_group_size)]
        torch.distributed.all_gather(grad_chunks, grad_output, group=xcd_group)
        grad_input = torch.cat(grad_chunks, dim=0)

        return grad_input



class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        return _DummyInpManipulation.apply(input_)
    


class _DummyInpManipulation2(torch.autograd.Function):
    """Dummy operator that will gather the outputs in forward and scatter in backward.
    This should be addressed by a gather_along_first_dim in the RowParallelLinear's forward pass.
    param_list and bucket_group to identify which params are None for that rank in the xcd group
    after we split grad_output -- to manage any issues with m-core/ddp wrapper that manages 
    param gather and reduction overlap in fwd/bwd passes respectively. 
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input_, param_list=None, bucket_group=None):
        """Forward with frozen weight."""
        xcd_group = parallel_state.get_xcd_intra_gpu_parallel_group()
        xcd_group_size = parallel_state.get_xcd_intra_gpu_parallel_world_size()
        xcd_rank = parallel_state.get_xcd_intra_gpu_parallel_rank()


        # input_ is a tuple (x, bias)

        input_chunks = [torch.zeros_like(input_[0]) for _ in range(xcd_group_size)]
        torch.distributed.all_gather(input_chunks, input_[0], group=xcd_group)
        input_rcs = torch.cat(input_chunks, dim=0)
        bias_rcs = None
        if input_[1] is not None:
            bias_chunks = [torch.zeros_like(input_[1]) for _ in range(xcd_group_size)]
            torch.distributed.all_gather(bias_chunks, input_[1], group=xcd_group)
            bias_rcs = torch.cat(bias_chunks, dim=0)

        output = (input_rcs, bias_rcs)

        # output, _ = gather_along_first_dim(input_, xcd_group, False)

        ctx.xcd_group = xcd_group
        ctx.xcd_rank = xcd_rank
        ctx.param_list = param_list
        ctx.bucket_group = bucket_group
        ctx.xcd_group_size = xcd_group_size
        # ctx.input_shape = input_.shape

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Backward to reconstruct the full grad."""
        #xcd_group = ctx.xcd_group
        xcd_group_size = ctx.xcd_group_size
        xcd_rank = parallel_state.get_xcd_intra_gpu_parallel_rank()
        param_list = ctx.param_list
        bucket_group = ctx.bucket_group
        # input_shape = ctx.input_shape

        grad_inp, grad_bias = grad_output

        grad_input_chunks = torch.chunk(grad_inp, xcd_group_size, dim=0)
        grad_bias_chunks = (
            torch.chunk(grad_bias, xcd_group_size, dim=0) if grad_bias is not None else None
        )
        
        grad_input = (grad_input_chunks[xcd_rank].contiguous(), 
                        grad_bias_chunks[xcd_rank].contiguous() if grad_bias_chunks is not None else None)

        if param_list is not None and bucket_group is not None:
            for i, param in enumerate(param_list):
                if i != xcd_rank:
                    bucket_group.register_grad_ready(param)
        #output_chunks = torch.chunk(grad_output, xcd_group_size, dim=0)
        #grad_input = output_chunks[xcd_rank].contiguous()
        #grad_chunks = [torch.zeros_like(grad_output) for _ in range(xcd_group_size)]
        #torch.distributed.all_gather(grad_chunks, grad_output, group=xcd_group)
        # grad_input = torch.cat(grad_chunks, dim=0)

        return grad_input, None, None


class DummyLayer2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        return _DummyInpManipulation2.apply(input_)