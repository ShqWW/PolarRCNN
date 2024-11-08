import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        if type(inp) == torch.Tensor:
            reduced_inp = inp.clone().cuda()  
        else:
            reduced_inp = torch.tensor([inp]).cuda()  
        dist.reduce(reduced_inp, dst=0)
        reduced_inp = reduced_inp/world_size
    return reduced_inp


def sum_tensor(inp):
    """
    Reduce the loss from all processes so that process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp.clone()
        dist.reduce(reduced_inp, dst=0)
        # dist.all_reduce(reduced_inp, op=dist.reduce_op.SUM)
        reduced_inp = reduced_inp
    return reduced_inp

def gather_tensor(inp):
    world_size = get_world_size()
    if world_size < 2:
        return [inp]
    with torch.no_grad():
        var_list = []
        gather_inp = inp.clone()
        dist.reduce(gather_inp, var_list)
    return var_list