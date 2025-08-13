import torch
import torch.nn as nn
from functools import partial
from megatron.core import parallel_state
from megatron.core.utils import is_torch_min_version

if is_torch_min_version("2.4.0a0"):
    custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
    custom_bwd = partial(torch.amp.custom_bwd, device_type="cuda")
else:
    custom_fwd = torch.cuda.amp.custom_fwd
    custom_bwd = torch.cuda.amp.custom_bwd


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
    
