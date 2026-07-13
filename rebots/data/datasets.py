import json
import random
from os import path

import numpy as np
import torch

from external.stable_audio_tools.src.data.dataset import (
    LocalDatasetConfig,
    get_latent_filenames,
)


class PreEncodedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs: list[LocalDatasetConfig],
        latent_crop_length=None,
        min_length_sec=None,
        max_length_sec=None,
        random_crop=False,
        latent_extension="npy",
        load_mean_sigma=False,
    ):
        super().__init__()
        self.filenames = []

        self.custom_metadata_fns = {}

        self.latent_extension = latent_extension
        self.load_mean_sigma = load_mean_sigma

        for config in configs:
            self.filenames.extend(get_latent_filenames(config.path, [latent_extension]))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = config.custom_metadata_fn

        mean_suffix = f"_mean.{latent_extension}"
        sigma_suffix = f"_sigma.{latent_extension}"
        if self.load_mean_sigma:
            mean_files = [filename for filename in self.filenames if filename.endswith(mean_suffix)]
            paired = []
            for mean_file in mean_files:
                base = mean_file[: -len(mean_suffix)]
                if path.exists(f"{base}{sigma_suffix}"):
                    paired.append(base)
            self.filenames = paired
        else:
            self.filenames = [
                filename
                for filename in self.filenames
                if not filename.endswith(mean_suffix) and not filename.endswith(sigma_suffix)
            ]

        self.latent_crop_length = latent_crop_length
        self.random_crop = random_crop

        self.min_length_sec = min_length_sec
        self.max_length_sec = max_length_sec

        print(f"Found {len(self.filenames)} files")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        latent_filename = self.filenames[idx]
        try:
            if self.load_mean_sigma:
                mean_filename = f"{latent_filename}_mean.{self.latent_extension}"
                sigma_filename = f"{latent_filename}_sigma.{self.latent_extension}"
                mean = torch.from_numpy(np.load(mean_filename))
                sigma = torch.from_numpy(np.load(sigma_filename))
                latents = mean + sigma * torch.randn_like(sigma)
                md_filename = f"{latent_filename}.json"
                info_latent_filename = f"{latent_filename}.{self.latent_extension}"
            else:
                latents = torch.from_numpy(np.load(latent_filename))  # [C, N]
                md_filename = latent_filename.replace(f".{self.latent_extension}", ".json")
                info_latent_filename = latent_filename

            with open(md_filename, "r") as f:
                try:
                    info = json.load(f)
                except:
                    raise Exception(f"Couldn't load metadata file {md_filename}")

            info["latent_filename"] = info_latent_filename

            if self.latent_crop_length is not None:

                # Get the last index from the padding mask, the index of the last 1 in the sequence
                last_ix = len(info["padding_mask"]) - 1 - info["padding_mask"][::-1].index(1)

                start = 0
                if self.random_crop and last_ix > self.latent_crop_length:
                    start = random.randint(0, last_ix - self.latent_crop_length)

                latents = latents[:, start : start + self.latent_crop_length]

                info["padding_mask"] = info["padding_mask"][start : start + self.latent_crop_length]

                info["latent_crop_length"] = self.latent_crop_length
                info["latent_crop_start"] = start

            info["padding_mask"] = [torch.tensor(info["padding_mask"])]

            seconds_total = info["seconds_total"]

            if self.min_length_sec is not None and seconds_total < self.min_length_sec:
                return self[random.randrange(len(self))]

            if self.max_length_sec is not None and seconds_total > self.max_length_sec:
                return self[random.randrange(len(self))]

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in latent_filename:
                    custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                    custom_metadata = custom_metadata_fn(info, None)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

                if "__replace__" in info and info["__replace__"] is not None:
                    # Replace the latents with the new latents if the custom metadata function returns a new set of latents
                    latents = info["__replace__"]

            info["audio"] = latents

            return (latents, info)
        except Exception as e:
            print(f"Couldn't load file {latent_filename}: {e}")
            return self[random.randrange(len(self))]
