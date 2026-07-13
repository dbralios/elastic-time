import torch


def load_stable_audio_open_checkpoint(ckpt_path: str, map_location="cpu") -> dict:
    """
    Load stable audio open checkpoint and extract autoencoder state dict.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        map_location (str, optional): Defaults to "cpu".

    Returns:
        dict: State dict of the autoencoder model
    """
    state_dict = torch.load(ckpt_path, map_location=map_location)
    keys = list(state_dict["state_dict"].keys())
    model_keys = [k for k in keys if k.startswith("pretransform")]
    autoencoder_state_dict = {k.replace("pretransform.model.", ""): state_dict["state_dict"][k] for k in model_keys}
    return autoencoder_state_dict


def get_checkpoint_loader_util_fn(name: str):
    """
    Get checkpoint loader utility function by name.

    Args:
        name (str): Name of the checkpoint loader utility function.

    Returns:
        Callable: Checkpoint loader utility function.
    """
    if name == "stable_audio_open":
        return load_stable_audio_open_checkpoint
    else:
        raise ValueError(f"Unknown checkpoint loader utility function: {name}")
