import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional
from warnings import warn

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from rebots import config, utils
from rebots.data.datasets import PreEncodedDataset
from rebots.models.components.codecslime_replace import mix_with_codecslime_segments
from rebots.models.components.elastic_time_replace import segment_matrix_to_segment_length_histogram
from rebots.utils.lr_schedulers import get_lr

log = utils._logging.get_logger("DEBUG")


class TrainingRecipeSingleDevice:
    """
    Full training recipe for a Codec-Slime style rebottleneck model.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = torch.device(cfg.device)
        self._dtype = utils.precision.get_dtype(cfg.dtype, device=self._device)

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._wandb_hist_every_n_steps = cfg.get("wandb_hist_every_n_steps", 100)

        # Training cfg
        self._gradient_accumulation_steps = 1
        if cfg.get("gradient_accumulation_steps", 1) != 1:
            raise NotImplementedError("Gradient accumulation is not supported for this recipe. Please set it to 1.")
        self._resume_from_checkpoint = cfg.get("resume_from_checkpoint", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._conditional_encoder = cfg.get("conditional_encoder", False)
        self._conditional_decoder = cfg.get("conditional_decoder", False)
        self._freeze_encoder = cfg.get("freeze_encoder", False)

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = cfg.get("seed", None)
        utils.set_seed(seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None))
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self.save_every_n_epochs = cfg.get("save_every_n_epochs", 3)

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """

        if cfg_checkpointer is None or not self._resume_from_checkpoint:
            return None

        state_dict: dict[str, Any] = {}

        ckpt_path = Path(cfg_checkpointer.get("model_checkpoint_path", ""))
        recipe_path = Path(cfg_checkpointer.get("recipe_checkpoint_path", ""))

        if ckpt_path.is_file():
            model_state_dict = torch.load(ckpt_path, map_location="cpu")
            state_dict[utils.MODEL_KEY] = model_state_dict
            log.info(f"Model checkpoint loaded from {ckpt_path}")

        if recipe_path.is_file():
            recipe_state_dict = torch.load(recipe_path, map_location="cpu")
            state_dict.update(recipe_state_dict)
            log.info(f"Recipe checkpoint loaded from {recipe_path}")

        return state_dict

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        """Updates the recipe state from checkpoint."""
        required_keys = [
            utils.SEED_KEY,
            utils.EPOCHS_KEY,
            utils.TOTAL_EPOCHS_KEY,
            utils.MAX_STEPS_KEY,
        ]
        for key in required_keys:
            if key not in ckpt_dict:
                raise KeyError(f"Key {key} not found in checkpoint dictionary.")

        self.epochs_run = ckpt_dict[utils.EPOCHS_KEY] + 1

        # Warn the user and prevent the override
        if self.seed != ckpt_dict[utils.SEED_KEY]:
            warn(
                message=(
                    "Config value for seed does not match the checkpoint value, "
                    f"using the checkpoint value: {ckpt_dict[utils.SEED_KEY]}"
                )
            )
            self.seed = ckpt_dict[utils.SEED_KEY]

        # Warn the user and prevent the override
        if self.max_steps_per_epoch != ckpt_dict[utils.MAX_STEPS_KEY]:
            warn(
                message=(
                    "Config value for max_steps_per_epoch does not match the checkpoint value, "
                    f"using the checkpoint value: {ckpt_dict[utils.MAX_STEPS_KEY]}"
                )
            )
            self.max_steps_per_epoch = ckpt_dict[utils.MAX_STEPS_KEY]

        # warn the user but allow the override
        if self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]:
            warn(
                message=(
                    "Config value for total_epochs does not match the checkpoint value, "
                    f"using the config value: {self.total_epochs}"
                )
            )

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up the recipe state correctly. This includes setting recipe attributes based
        on the ``resume_from_checkpoint`` flag.
        """

        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        # Load checkpoint if resuming from checkpoint
        ckpt_dict = self.load_checkpoint(cfg.get("checkpointer", None))

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._compile = cfg.get("compile", False)
        self._compile_mode = cfg.get("compile_mode", "default")
        self._compile_dynamic = cfg.get("compile_dynamic", False)
        self._compile_fullgraph = cfg.get("compile_fullgraph", False)
        self._compile_discriminator = cfg.get("compile_discriminator", False)

        self._autoencoder = self._setup_autoencoder(
            cfg_autoencoder=cfg.autoencoder,
        )

        self._model = self._setup_model(
            cfg_model=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY] if ckpt_dict else None,
        )

        if self._freeze_encoder:
            if not hasattr(self._model, "encoder"):
                raise ValueError("freeze_encoder=True requires model to define an encoder module.")
            for param in self._model.encoder.parameters():
                param.requires_grad = False
            log.info("Model encoder parameters are frozen (requires_grad=False).")

        self._discriminator = self._setup_discriminator(
            cfg_discriminator=cfg.get("discriminator", None),
            state_dict=(
                ckpt_dict[utils.DISCRIMINATOR_KEY] if ckpt_dict and utils.DISCRIMINATOR_KEY in ckpt_dict else None
            ),
        )

        self._model_encode, self._model_decode = self._setup_model_ops(self._model)
        self._discriminator_forward = self._setup_discriminator_ops(self._discriminator)

        autoencoder_params = sum(p.numel() for p in self._autoencoder.parameters())
        model_params = sum(p.numel() for p in self._model.parameters())
        log.info(f"Autoencoder parameters: {autoencoder_params:,}")
        log.info(f"Model parameters: {model_params:,}")
        if self._discriminator is not None:
            discriminator_params = sum(p.numel() for p in self._discriminator.parameters())
            log.info(f"Discriminator parameters: {discriminator_params:,}")

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(ckpt_dict[utils.OPT_KEY] if ckpt_dict and utils.OPT_KEY in ckpt_dict else None),
            params=self._model.parameters(),
        )

        if self._discriminator is not None:
            assert cfg.get("optimizer_discriminator", None) is not None
            self._optimizer_d = self._setup_optimizer(
                cfg_optimizer=cfg.optimizer_discriminator,
                params=self._discriminator.parameters(),
                opt_state_dict=(ckpt_dict[utils.OPT_D_KEY] if ckpt_dict and utils.OPT_D_KEY in ckpt_dict else None),
            )

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        if self._discriminator is not None:
            self._loss_gan = config.instantiate(cfg.loss_gan)

        log.info("Model is initialized.")

        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        if self._resume_from_checkpoint:
            if ckpt_dict is None:
                warn(message=("resume_from_checkpoint is set to True but no checkpoint was found. "))
            else:
                self._update_recipe_state(ckpt_dict=ckpt_dict)

        self._steps_per_epoch = len(self._dataloader) // self._gradient_accumulation_steps
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < self._steps_per_epoch:
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        self._lr_scheduler = self._setup_lr_scheduler(
            optimizer=self._optimizer,
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        if self._discriminator is not None:
            self._lr_scheduler_d = self._setup_lr_scheduler(
                optimizer=self._optimizer_d,
                cfg_lr_scheduler=cfg.get("lr_scheduler_discriminator", None),
                num_training_steps=self.total_epochs * self._steps_per_epoch,
                last_epoch=self.global_step - 1,
            )

        self._loss_weights = cfg.get("loss_weights", None)

        self._masker = config.instantiate(cfg.masker)
        self._max_segment_length = int(getattr(self._masker, "max_segment_length", 1))

    def _setup_autoencoder(
        self,
        cfg_autoencoder: DictConfig,
    ) -> nn.Module:
        """
        Set up the autoencoder model.
        """

        ckpt_path = cfg_autoencoder.pop("checkpoint_path", None)
        ckpt_loader_util_fn_cfg = cfg_autoencoder.pop("checkpoint_loader_util_fn_cfg", None)

        autoencoder_state_dict = None
        if ckpt_path is not None:
            if ckpt_loader_util_fn_cfg is not None:
                ckpt_loader_util_fn = config.instantiate(ckpt_loader_util_fn_cfg)
                autoencoder_state_dict = ckpt_loader_util_fn(ckpt_path, map_location="cpu")
            else:
                autoencoder_state_dict = torch.load(ckpt_path, map_location="cpu")

        with self._device:
            autoencoder = config.instantiate(cfg_autoencoder)

        if autoencoder_state_dict is not None:
            missing, unexpected = autoencoder.load_state_dict(autoencoder_state_dict, strict=False)
            log.info(f"Autoencoder loaded from {ckpt_path}")
            log.info(f"Missing keys: {missing}")
            log.info(f"Unexpected keys: {unexpected}")

        log.info("Autoencoder is initialized.")
        return autoencoder

    def _setup_model(
        self,
        cfg_model: DictConfig,
        model_state_dict: dict[str, Any] = None,
    ) -> nn.Module:
        """
        Set up the model.
        """
        ckpt_path = cfg_model.pop("checkpoint_path", None)
        ckpt_loader_util_fn_cfg = cfg_model.pop("checkpoint_loader_util_fn_cfg", None)

        model_init_state_dict = None
        if model_state_dict is None and ckpt_path is not None:
            if ckpt_loader_util_fn_cfg is not None:
                ckpt_loader_util_fn = config.instantiate(ckpt_loader_util_fn_cfg)
                model_init_state_dict = ckpt_loader_util_fn(ckpt_path, map_location="cpu")
            else:
                model_init_state_dict = torch.load(ckpt_path, map_location="cpu")

            if isinstance(model_init_state_dict, dict) and utils.MODEL_KEY in model_init_state_dict:
                model_init_state_dict = model_init_state_dict[utils.MODEL_KEY]

        with self._device, utils.set_default_dtype(self._dtype):
            model = config.instantiate(cfg_model)

            if model_state_dict is not None:
                missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
                log.info("Model loaded from checkpoint")
                log.info(f"Missing keys: {missing}")
                log.info(f"Unexpected keys: {unexpected}")
            elif model_init_state_dict is not None:
                missing, unexpected = model.load_state_dict(model_init_state_dict, strict=False)
                log.info(f"Model initialized from {ckpt_path}")
                log.info(f"Missing keys: {missing}")
                log.info(f"Unexpected keys: {unexpected}")

            model = model.to(self._device)

        log.info(f"Model is initialized with precision {self._dtype}.")

        return model

    def _maybe_compile_callable(self, fn: Callable[..., Any], fn_name: str) -> Callable[..., Any]:
        if not self._compile:
            return fn

        log.info(
            f"Compiling {fn_name} with torch.compile "
            f"(mode={self._compile_mode}, dynamic={self._compile_dynamic}, fullgraph={self._compile_fullgraph})."
        )
        return torch.compile(
            fn,
            mode=self._compile_mode,
            dynamic=self._compile_dynamic,
            fullgraph=self._compile_fullgraph,
        )

    def _setup_model_ops(self, model: nn.Module) -> tuple[Callable[..., Any], Callable[..., Any]]:
        encode_fn = self._maybe_compile_callable(model.encode, "model.encode")
        decode_fn = self._maybe_compile_callable(model.decode, "model.decode")
        return encode_fn, decode_fn

    def _setup_discriminator_ops(self, discriminator: Optional[nn.Module]) -> Optional[Callable[..., Any]]:
        if discriminator is None:
            return None
        if not self._compile or not self._compile_discriminator:
            return discriminator
        return self._maybe_compile_callable(discriminator, "discriminator")

    def _setup_discriminator(
        self,
        cfg_discriminator: DictConfig,
        state_dict: dict[str, Any] = None,
    ) -> nn.Module:
        """
        Set up the discriminator model.
        """
        if cfg_discriminator is None:
            return None

        with self._device, utils.set_default_dtype(self._dtype):
            model = config.instantiate(cfg_discriminator)

            if state_dict is not None:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                log.info("Discriminator loaded from checkpoint")
                log.info(f"Missing keys: {missing}")
                log.info(f"Unexpected keys: {unexpected}")

        log.info(f"Discriminator model is initialized with precision {self._dtype}.")

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        opt_state_dict: Optional[dict[str, Any]] = None,
        params: Generator[nn.Parameter, None, None] = None,
    ) -> Optional[Optimizer]:
        """
        Set up the optimizer. This method also handles loading the optimizer state_dict, if specified.
        """

        optimizer = config.instantiate(cfg_optimizer, params)

        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)
        log.info("Optimizer is initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        optimizer: Optimizer,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        """
        Set up the learning rate scheduler.
        """
        if cfg_lr_scheduler is None:
            log.info("No learning rate scheduler configured. Using constant learning rate.")
            return None

        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        num_workers: int = 4,
    ) -> tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here.
        """

        configs = [config.instantiate(cur_cfg) for cur_cfg in cfg_dataset["configs"]]
        cfg_dataset.pop("configs")

        ds = config.instantiate(cfg_dataset, configs=configs)
        self._use_preencoded_latents = isinstance(ds, PreEncodedDataset)

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
        )

        log.info("Dataset and Sampler are initialized.")
        if self._use_preencoded_latents:
            log.info("Using pre-encoded latents dataset.")

        return sampler, dataloader

    def _setup_metrics(
        self,
        cfg_metrics: DictConfig,
    ) -> dict[str, Any]:
        """
        Set up the metrics.
        """

        metrics = {}
        for cur_cfg_metric in cfg_metrics:
            cur_metric_name = cur_cfg_metric.pop("name", None)

            cur_metric = config.instantiate(cur_cfg_metric)
            metrics[cur_metric_name] = cur_metric
            cur_metric.to(self._device)

            log.info(f"Metric {cur_metric_name} is initialized.")

        self._metrics = metrics
        log.info("Metrics are initialized.")

        return self._metrics

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save state dict to file.
        """
        ckpt_dict = {
            utils.MODEL_KEY: self._model.state_dict(),
            utils.DISCRIMINATOR_KEY: self._discriminator.state_dict() if self._discriminator is not None else None,
        }

        if epoch + 1 < self.total_epochs:
            ckpt_dict.update(
                {
                    utils.SEED_KEY: self.seed,
                    utils.EPOCHS_KEY: self.epochs_run,
                    utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                    utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )
            ckpt_dict[utils.OPT_KEY] = self._optimizer.state_dict()

        intermediate_checkpoint = epoch + 1 < self.total_epochs

        output_path = Path.joinpath(
            Path(self._output_dir),
            f"ckpt_epoch_{epoch}",
        ).with_suffix(".pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt_dict[utils.MODEL_KEY], output_path)

        log.info(f"Model checkpoint of size {os.path.getsize(output_path) / 1024**3:.2f} GiB saved to {output_path}")

        if intermediate_checkpoint:
            _ = ckpt_dict.pop(utils.MODEL_KEY, None)
            output_path = Path.joinpath(Path(self._output_dir), "recipe_state.pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt_dict, output_path)
            log.info(
                f"Recipe checkpoint of size {os.path.getsize(output_path) / 1024**3:.2f} GiB saved to {output_path}"
            )

    def train(self) -> None:
        """
        The core training loop.
        """
        if self._compile:
            target_modules = "model encode/decode"
            if self._discriminator is not None and self._compile_discriminator:
                target_modules += " and discriminator"
            log.info(f"NOTE: torch.compile is enabled for {target_modules}. Expect a relatively slow first iteration.")

        self._autoencoder.eval()

        self._model.train()
        if self._freeze_encoder:
            self._model.encoder.eval()
        if self._discriminator is not None:
            self._discriminator.train()

        if self._metric_logger is not None and hasattr(self._metric_logger, "watch"):
            log.info("Watching model with metric logger.")
            self._metric_logger.watch(self._model)

        self._optimizer.zero_grad()

        t0 = time.perf_counter()
        running_loss = 0
        running_loss_dict = {}
        for key in self._loss_weights.keys():
            running_loss_dict[key] = 0.0

        for curr_epoch in range(self.epochs_run, self.total_epochs):
            self._sampler.set_epoch(curr_epoch)

            pbar = tqdm(total=self._steps_per_epoch)
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps) == self.max_steps_per_epoch
                ):
                    break

                if self._use_preencoded_latents:
                    latents, _ = batch[0], batch[1]
                    latents = latents.to(self._device, dtype=self._dtype)
                    latent_targets = latents.clone().detach()
                else:
                    waveforms, _ = batch[0], batch[1]
                    waveforms = waveforms.to(self._device, dtype=self._dtype)

                    autoencoder_inputs = waveforms
                    with torch.no_grad():
                        latent_representations, _ = self._autoencoder.encode(autoencoder_inputs, return_info=True)
                    latents = latent_representations
                    latent_targets = latent_representations.clone().detach()

                mask_sample = self._masker.sample(
                    latents.shape[0],
                    latents.shape[-1],
                    latents.device,
                    global_step=self.global_step,
                )
                cond_input = mask_sample.target_replace_fraction

                if self._conditional_encoder:
                    z, info = self._model_encode(latents, return_info=True, cond_input=cond_input)
                else:
                    z, info = self._model_encode(latents, return_info=True)

                h = self._masker.replace(z.detach(), mask_sample)
                z_dec = mix_with_codecslime_segments(z, h)

                if self._conditional_decoder:
                    pred_latents = self._model_decode(z_dec, cond_input=cond_input)
                else:
                    pred_latents = self._model_decode(z_dec)

                loss = {}
                loss_d = {}

                loss["loss_reconstruction"] = self._loss_fn(pred_latents, latent_targets)
                loss["loss_kl"] = info.get("kl", torch.zeros(1, device=self._device))
                loss["loss_commitment"] = info.get("commitment_loss", torch.zeros(1, device=self._device)).mean()
                loss["loss_codebook"] = info.get("codebook_loss", torch.zeros(1, device=self._device)).mean()

                if self._discriminator is not None:
                    d_real, fm_real = self._discriminator_forward(latent_targets)
                    d_fake, fm_fake = self._discriminator_forward(pred_latents.clone().detach())

                    loss_d["loss_discriminator"] = self._loss_gan.discriminator_loss(d_fake, d_real)

                    current_loss_d = 0.0
                    for key in loss_d.keys():
                        if self._loss_weights.get(key, None) is not None:
                            running_loss_dict[key] += loss_d[key].item()
                            current_loss_d += self._loss_weights[key] * loss_d[key]

                    self._optimizer_d.zero_grad(set_to_none=True)
                    current_loss_d.backward()

                    if self._clip_grad_norm is not None:
                        grad_norm_d = torch.nn.utils.clip_grad_norm_(
                            self._discriminator.parameters(),
                            max_norm=float(self._clip_grad_norm),
                        )
                    self._optimizer_d.step()

                if self._discriminator is not None:
                    d_real, fm_real = self._discriminator_forward(latent_targets)
                    d_fake, fm_fake = self._discriminator_forward(pred_latents)

                    loss["loss_adversarial"] = self._loss_gan.generator_loss(d_fake, d_real)
                    loss["loss_feature_matching"] = self._loss_gan.feature_matching_loss(fm_fake, fm_real)

                current_loss = 0.0
                for key in loss.keys():
                    if self._loss_weights.get(key, None) is not None:
                        running_loss_dict[key] += loss[key].item()
                        current_loss += self._loss_weights[key] * loss[key]

                running_loss += current_loss

                current_loss.backward()

                if self._clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        max_norm=float(self._clip_grad_norm),
                    )
                self._optimizer.step()
                self._optimizer.zero_grad(set_to_none=True)

                if self._lr_scheduler is not None:
                    self._lr_scheduler.step()
                if self._discriminator is not None and self._lr_scheduler_d is not None:
                    self._lr_scheduler_d.step()
                self.global_step += 1

                loss_to_log = running_loss.item()
                loss_to_log_dict = {}
                for key in self._loss_weights.keys():
                    loss_to_log_dict[key] = running_loss_dict[key] / self._gradient_accumulation_steps

                pbar.update(1)
                pbar.set_description(f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}")

                if self.global_step % self._log_every_n_steps == 0:
                    time_per_step = time.perf_counter() - t0
                    log_dict = {
                        "loss": loss_to_log,
                        "lr": get_lr(self._optimizer),
                        "time_per_step": time_per_step,
                        "stats/latent_std": torch.std(z).item(),
                        "stats/target_replace_fraction_mean": mask_sample.target_replace_fraction.mean().item(),
                        "stats/realized_keep_ratio": (h == 0).float().mean().item(),
                    }

                    for key in loss_to_log_dict.keys():
                        log_dict[key] = loss_to_log_dict[key]

                    if (
                        hasattr(self._metric_logger, "_wandb")
                        and self._metric_logger._wandb.run
                        and self._wandb_hist_every_n_steps > 0
                        and self.global_step % self._wandb_hist_every_n_steps == 0
                    ):
                        wandb = self._metric_logger._wandb

                        h_bin_avg = segment_matrix_to_segment_length_histogram(
                            h.detach().to(dtype=torch.long), K=max(self._max_segment_length - 1, 0)
                        )
                        bin_rows = [[k, value] for k, value in enumerate(h_bin_avg.tolist())]
                        bin_table = wandb.Table(data=bin_rows, columns=["bin", "avg_count"])
                        log_dict["stats/h_bin_plot"] = wandb.plot.bar(
                            bin_table,
                            "bin",
                            "avg_count",
                            title="Codec-Slime Offset Bin Avg Count",
                        )

                    if self._clip_grad_norm is not None:
                        log_dict.update({"grad_norm": grad_norm})
                        if self._discriminator is not None:
                            log_dict.update({"grad_norm_d": grad_norm_d})
                    self._metric_logger.log_dict(
                        log_dict,
                        step=self.global_step,
                    )

                running_loss = 0
                for key in self._loss_weights.keys():
                    running_loss_dict[key] = 0.0
                t0 = time.perf_counter()

            if self.epochs_run % self.save_every_n_epochs == 0 or (curr_epoch + 1) == self.total_epochs:
                self.save_checkpoint(epoch=curr_epoch)
            self.epochs_run += 1

    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="TrainingRecipeSingleDevice", cfg=cfg)
    recipe = TrainingRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
