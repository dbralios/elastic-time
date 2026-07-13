import torch
from safetensors.torch import load_file

from torch.nn.utils import remove_weight_norm


def load_ckpt_state_dict(ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)[
            "state_dict"
        ]

    return state_dict


def remove_weight_norm_from_model(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            print(f"Removing weight norm from {module}")
            remove_weight_norm(module)

    return model


# Get torch.compile flag from environment variable ENABLE_TORCH_COMPILE

import os

enable_torch_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"


def compile(function, *args, **kwargs):

    if enable_torch_compile:
        try:
            return torch.compile(function, *args, **kwargs)
        except RuntimeError:
            return function

    return function
