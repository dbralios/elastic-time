# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from itertools import chain
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from ._logging import get_logger
from ._device import get_device

_log: logging.Logger = get_logger()


def is_distributed() -> bool:
    """Check if all environment variables required to initialize torch.distributed are set
    and distributed is properly installed. This indicates a distributed run.
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization

    Checks the following conditions:

    * torch.distributed is available
    * master port and master address environment variables are set
    * world size is >1
    * rank environment variable is set

    Returns:
        bool: True if all of the above conditions hold, False otherwise.
    """
    port = os.environ.get("MASTER_PORT", "")
    addr = os.environ.get("MASTER_ADDR", "")
    size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", -1))
    avlb = dist.is_available()
    return bool(port and addr and size >= 1 and rank >= 0 and avlb)


def _broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcasts a tensor from a source to all other processes.

    Args:
        tensor (torch.Tensor): torch.Tensor to broadcast.
        src (int, optional): Source rank. Defaults to 0.

    Returns:
        torch.Tensor: Broadcasted tensor.
    """
    if dist.is_available() and dist.is_initialized():
        device = tensor.device
        if dist.get_backend() == "nccl":
            tensor = tensor.to(get_device("cuda"))
        dist.broadcast(tensor, src=src, group=None)
        return tensor.to(device)
    else:
        return tensor


def get_distributed_backend(device_type: str, offload_ops_to_cpu: bool = False) -> str:
    """Gets the PyTorch Distributed backend based on device type.

    Args:
        device_type (str): Device type to get backend for.
        offload_ops_to_cpu (bool, optional): Flag to check if any operations should be offloaded to CPU.
            Examples of these kinds of operations are CPU offload for FSDP and asynchronous save for distributed
            checkpointing. Defaults to False.

    Example:
        >>> get_distributed_backend("cuda")
        'nccl'
        >>> get_distributed_backend("cpu")
        'gloo'
        >>> get_distributed_backend("cuda", offload_ops_to_cpu=True)
        'cuda:nccl,cpu:gloo'

    Returns:
        str: Distributed backend for use in ``torch.distributed.init_process_group``.
    """
    default_device_backend_map = dist.Backend.default_device_backend_map
    backend = "nccl"
    if device_type in default_device_backend_map:
        backend = default_device_backend_map[device_type]
    if offload_ops_to_cpu:
        backend = f"{device_type}:{backend},cpu:gloo"
    return backend


# @deprecated(
#     msg="The functionality of `init_distributed` is covered by `torch.distributed.init_process_group`. "
# )
# def init_distributed(**kwargs: Dict[str, Any]) -> bool:
#     """Initialize process group required for ``torch.distributed``.

#     Args:
#         **kwargs (Dict[str, Any]): Additional arguments to pass to torch.distributed.init_process_group.

#     Returns:
#         bool: True if torch.distributed is initialized.

#     Raises:
#         RuntimeError: If torch.distributed is already initialized.
#     """
#     if is_distributed():
#         if dist.is_initialized():
#             raise RuntimeError("torch.distributed already initialized.")
#         dist.init_process_group(**kwargs)
#         return True
#     else:
#         return False


def set_torch_num_threads() -> None:
    """
    Sets the number of threads used by torch to utilize all physical CPU
    cores for intra-op parallelism. Currently, this function sets num_threads
    to be the number of physical CPU cores divided by the number of GPUs as we
    use one process per GPU, and this avoids CPU oversubscription. Note that this is
    currently a rough approximation, and doesn't take into account environments where
    things like CPU affinity is set.
    """
    num_threads = os.cpu_count() // (
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )
    torch.set_num_threads(num_threads)
    _log.info(f"Set intra op parallelism no. of threads to {num_threads}")


def validate_no_params_on_meta_device(model: nn.Module) -> None:
    """
    Utility to validate that model has no params or buffers on meta device.
    If a meta param or buffer is found, an error indicating the param name will
    be raised.

    Args:
        model (nn.Module): model to check for meta params

    Raises:
        RuntimeError: If meta params or buffers exist in model
    """
    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")
