import os
import sys
import time
from pathlib import Path
from typing import Any, Generator, Optional
from warnings import warn

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from rebots import config, utils
from rebots.data.datasets import PreEncodedDataset
from rebots.utils.lr_schedulers import get_lr

log = utils._logging.get_logger("DEBUG")


class TrainingRecipeSingleDevice:
    """
    Full training recipe for a rebottleneck model on a single device.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = torch.device(cfg.device)
        self._dtype = utils.precision.get_dtype(cfg.dtype, device=self._device)

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)

        # Training cfg
        # self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = 1
        if cfg.get("gradient_accumulation_steps", 1) != 1:
            raise NotImplementedError("Gradient accumulation is not supported for this recipe. Please set it to 1.")
        self._resume_from_checkpoint = cfg.get("resume_from_checkpoint", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = cfg.get("seed", None)
        utils.set_seed(seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None))
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

        self._do_validation = False
        self._validation_max_steps = None
        self._use_preencoded_latents_train = False
        self._use_preencoded_latents_validation = False
        self._metrics_dict = {}

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
        """
        Updates the recipe state from checkpoint.
        """
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

        if self.seed != ckpt_dict[utils.SEED_KEY]:
            warn(
                message=(
                    "Config value for seed does not match the checkpoint value, "
                    f"using the checkpoint value: {ckpt_dict[utils.SEED_KEY]}"
                )
            )
            self.seed = ckpt_dict[utils.SEED_KEY]

        if self.max_steps_per_epoch != ckpt_dict[utils.MAX_STEPS_KEY]:
            warn(
                message=(
                    "Config value for max_steps_per_epoch does not match the checkpoint value, "
                    f"using the checkpoint value: {ckpt_dict[utils.MAX_STEPS_KEY]}"
                )
            )
            self.max_steps_per_epoch = ckpt_dict[utils.MAX_STEPS_KEY]

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

        ckpt_dict = self.load_checkpoint(cfg.get("checkpointer", None))

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._compile = cfg.compile

        self._autoencoder = self._setup_autoencoder(
            cfg_autoencoder=cfg.autoencoder,
        )

        self._model = self._setup_model(
            cfg_model=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY] if ckpt_dict else None,
        )

        self._discriminator = self._setup_discriminator(
            cfg_discriminator=cfg.get("discriminator", None),
            state_dict=(
                ckpt_dict[utils.DISCRIMINATOR_KEY] if ckpt_dict and utils.DISCRIMINATOR_KEY in ckpt_dict else None
            ),
        )

        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            # opt_state_dict=(
            #     ckpt_dict[training.OPT_KEY] if self._resume_from_checkpoint else None
            # ),
            params=self._model.parameters(),
            opt_state_dict=(ckpt_dict[utils.OPT_KEY] if ckpt_dict and utils.OPT_KEY in ckpt_dict else None),
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

        # if self._compile:
        #     training.compile_loss(self._loss_fn)

        log.info("Model is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._sampler, self._dataloader, self._use_preencoded_latents_train = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        cfg_dataset_validation = cfg.get("dataset_validation", None)
        self._do_validation = cfg_dataset_validation is not None
        self._validation_max_steps = cfg.get("validation_max_steps", None)
        if cfg_dataset_validation is not None:
            validation_batch_size = cfg.get("validation_batch_size", 1)
            (
                self._sampler_validation,
                self._dataloader_validation,
                self._use_preencoded_latents_validation,
            ) = self._setup_data(
                cfg_dataset=cfg_dataset_validation,
                shuffle=False,
                batch_size=validation_batch_size,
                drop_last=False,
            )

            cfg_metrics = cfg.get("metrics", None)
            if cfg_metrics is None:
                raise ValueError("dataset_validation is set, but no `metrics` were configured in OmegaConf.")
            self._metrics_dict = self._setup_metrics(cfg_metrics=cfg_metrics)

        if self._resume_from_checkpoint:
            if ckpt_dict is None:
                warn(message=("resume_from_checkpoint is set to True but no checkpoint was found. "))
            else:
                self._update_recipe_state(ckpt_dict=ckpt_dict)

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        #
        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader, the max_steps_per_epoch param set by the user and the
        # gradient_accumulation_steps param. This value is used for logging and tracking
        # training state. The computation should happen after the dataloader has been setup
        self._steps_per_epoch = len(self._dataloader) // self._gradient_accumulation_steps
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < self._steps_per_epoch:
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        # Setup lr scheduler
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

        # Setup loss weights
        self._loss_weights = cfg.get("loss_weights", None)

    def _setup_autoencoder(
        self,
        cfg_autoencoder: DictConfig,
    ) -> nn.Module:
        """
        Set up the autoencoder model.
        """

        ckpt_path = cfg_autoencoder.pop("checkpoint_path", None)
        autoencoder_state_dict = None
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            keys = list(state_dict["state_dict"].keys())
            model_keys = [k for k in keys if k.startswith("pretransform")]
            autoencoder_state_dict = {
                k.replace("pretransform.model.", ""): state_dict["state_dict"][k] for k in model_keys
            }

        with self._device:
            autoencoder = config.instantiate(cfg_autoencoder)

        if autoencoder_state_dict is not None:
            autoencoder.load_state_dict(autoencoder_state_dict)
            log.info(f"Autoencoder loaded from {ckpt_path}")

        log.info("Autoencoder is initialized.")
        return autoencoder

    def _setup_model(
        self,
        cfg_model: DictConfig,
        model_state_dict: dict[str, Any] = None,
    ) -> nn.Module:
        """
        Set up the model including enabling activation checkpointing.
        """
        with self._device, utils.set_default_dtype(self._dtype):
            model = config.instantiate(cfg_model)

            if model_state_dict is not None:
                missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
                log.info("Model loaded from checkpoint")
                log.info(f"Missing keys: {missing}")
                log.info(f"Unexpected keys: {unexpected}")

            model = model.to(self._device)

        # if compile_model:
        #     training.compile_model(model)

        # model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        # training.validate_expected_param_dtype(
        #     model.named_parameters(), dtype=self._dtype
        # )

        log.info(f"Model is initialized with precision {self._dtype}.")

        return model

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
        Set up the learning rate scheduler based on the provided configuration.
        It handles both standard optimization and optimizer-in-backward cases, and supports
        schedulers from both torchtune.modules and torch.optim.

        Args:
            optimizer (Optimizer): The optimizer to be used.
            cfg_lr_scheduler (Optional[DictConfig]): The learning rate scheduler configuration.
            num_training_steps (int): The total number of training steps.
            last_epoch (int): The index of the last epoch.

        Returns:
            lr_scheduler (Optional[Optimizer]): The learning rate scheduler.
        """
        if cfg_lr_scheduler is None:
            log.info("No learning rate scheduler configured. Using constant learning rate.")
            return None

        # Instantiate the learning rate scheduler
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
        drop_last: bool = True,
    ) -> tuple[DistributedSampler, DataLoader, bool]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """

        cfg_dataset_local = OmegaConf.create(OmegaConf.to_container(cfg_dataset, resolve=False))
        self.sample_rate = cfg_dataset_local.get("sample_rate", None)

        configs = [config.instantiate(cur_cfg) for cur_cfg in cfg_dataset_local["configs"]]
        cfg_dataset_local.pop("configs")

        ds = config.instantiate(cfg_dataset_local, configs=configs)
        use_preencoded_latents = isinstance(ds, PreEncodedDataset)

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
            drop_last=drop_last,
            num_workers=num_workers,
        )

        log.info("Dataset and Sampler are initialized.")
        if use_preencoded_latents:
            log.info("Using pre-encoded latents dataset.")

        return sampler, dataloader, use_preencoded_latents

    def _setup_metrics(
        self,
        cfg_metrics: DictConfig,
    ) -> dict[str, Any]:
        """
        Set up the metrics.
        """

        metrics = {}
        for cur_cfg_metric in cfg_metrics:
            cur_cfg_metric_local = OmegaConf.create(OmegaConf.to_container(cur_cfg_metric, resolve=False))
            cur_metric_name = cur_cfg_metric_local.pop("name", None)
            if cur_metric_name is None:
                raise ValueError("Each metric config must define a `name`.")

            cur_metric = config.instantiate(cur_cfg_metric_local)
            metrics[cur_metric_name] = cur_metric
            if hasattr(cur_metric, "to"):
                cur_metric.to(self._device)

            log.info(f"Metric {cur_metric_name} is initialized.")

        self._metrics = metrics
        log.info("Metrics are initialized.")

        return self._metrics

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save state dict to file. The recipe save_checkpoint method is responsible for
        correctly creating the checkpoint dict and passing to the checkpointer.
        """
        ckpt_dict = {
            utils.MODEL_KEY: self._model.state_dict(),
            utils.DISCRIMINATOR_KEY: self._discriminator.state_dict() if self._discriminator is not None else None,
        }
        # if training is in-progress, checkpoint the optimizer state as well
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
            if self._discriminator is not None:
                ckpt_dict[utils.OPT_D_KEY] = self._optimizer_d.state_dict()

        intermediate_checkpoint = epoch + 1 < self.total_epochs

        output_path = Path.joinpath(
            Path(self._output_dir),
            f"ckpt_epoch_{epoch}",
        ).with_suffix(".pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt_dict[utils.MODEL_KEY], output_path)

        log.info(f"Model checkpoint of size {os.path.getsize(output_path) / 1024**3:.2f} GiB saved to {output_path}")

        # If the recipe state needs to be output, first remove the model state dict
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
        The core training loop. Supports training on subsets of the dataset using the
        ``max_steps_per_epoch``.
        """
        # if self._compile:
        #     log.info(
        #         "NOTE: torch.compile is enabled and model is compiled in first forward. Expect a relatively slow first iteration."
        #     )

        # autoencoder is not trained, so we set it to eval mode
        self._autoencoder.eval()

        # model is trained, so we set it to train mode
        self._model.train()
        if self._discriminator is not None:
            self._discriminator.train()

        # Watch the model parameters and gradients
        if self._metric_logger is not None and hasattr(self._metric_logger, "watch"):
            log.info("Watching model with metric logger.")
            self._metric_logger.watch(self._model)

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        running_loss_dict = {}
        for key in self._loss_weights.keys():
            running_loss_dict[key] = 0.0

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            pbar = tqdm(total=self._steps_per_epoch)
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps) == self.max_steps_per_epoch
                ):
                    break

                if self._use_preencoded_latents_train:
                    latents, metadata = batch[0], batch[1]
                    latents = latents.to(self._device, dtype=self._dtype)
                    latent_targets = latents.clone().detach()
                else:
                    waveforms, metadata = batch[0], batch[1]
                    waveforms = waveforms.to(self._device, dtype=self._dtype)

                    # Concat in the batch dimension
                    autoencoder_inputs = waveforms

                    # Encode the waveforms to latent representations
                    with torch.no_grad():
                        latent_representations, _ = self._autoencoder.encode(autoencoder_inputs, return_info=True)

                    latents = latent_representations
                    latent_targets = latent_representations.clone().detach()

                # Feed the latents to the model
                pred_latents, info = self._model(latents, return_info=True)

                # Compute loss
                loss = {}
                loss_d = {}

                # Reconstruction loss
                loss["loss_reconstruction"] = self._loss_fn(pred_latents, latent_targets)

                # KL loss
                loss["loss_kl"] = info.get("kl", torch.zeros(1, device=self._device))
                # VQ losses
                loss["loss_commitment"] = info.get("commitment_loss", torch.zeros(1, device=self._device)).mean()
                loss["loss_codebook"] = info.get("codebook_loss", torch.zeros(1, device=self._device)).mean()

                # Discriminator training
                if self._discriminator is not None:
                    d_real, fm_real = self._discriminator(latent_targets)
                    d_fake, fm_fake = self._discriminator(pred_latents.clone().detach())

                    loss_d["loss_discriminator"] = self._loss_gan.discriminator_loss(d_fake, d_real)

                    current_loss_d = 0.0
                    for key in loss_d.keys():
                        if self._loss_weights.get(key, None) is not None:
                            running_loss_dict[key] += loss_d[key].item()
                            current_loss_d += self._loss_weights[key] * loss_d[key]

                    self._optimizer_d.zero_grad(set_to_none=True)
                    current_loss_d.backward()

                    # Step with optimizer
                    if self._clip_grad_norm is not None:
                        grad_norm_d = torch.nn.utils.clip_grad_norm_(
                            self._discriminator.parameters(),
                            max_norm=float(self._clip_grad_norm),
                        )
                    self._optimizer_d.step()
                    # self._optimizer_d.zero_grad(
                    #     set_to_none=True
                    # )  # Unecessary, since the other optimizer doesn't update discr params

                # Generator training
                if self._discriminator is not None:
                    d_real, fm_real = self._discriminator(latent_targets)
                    d_fake, fm_fake = self._discriminator(pred_latents)

                    loss["loss_adversarial"] = self._loss_gan.generator_loss(d_fake, d_real)
                    loss["loss_feature_matching"] = self._loss_gan.feature_matching_loss(fm_fake, fm_real)

                current_loss = 0.0
                for key in loss.keys():
                    if self._loss_weights.get(key, None) is not None:
                        running_loss_dict[key] += loss[key].item()
                        current_loss += self._loss_weights[key] * loss[key]

                running_loss += current_loss

                current_loss.backward()

                # Step with optimizer
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

                # Log per-step metrics
                if self.global_step % self._log_every_n_steps == 0:
                    time_per_step = time.perf_counter() - t0
                    log_dict = {
                        "loss": loss_to_log,
                        "lr": get_lr(self._optimizer),
                        "time_per_step": time_per_step,
                    }
                    for key in loss_to_log_dict.keys():
                        log_dict[key] = loss_to_log_dict[key]

                    if self._clip_grad_norm is not None:
                        log_dict.update({"grad_norm": grad_norm})
                        if self._discriminator is not None:
                            log_dict.update({"grad_norm_d": grad_norm_d})
                    self._metric_logger.log_dict(
                        log_dict,
                        step=self.global_step,
                    )

                # Reset running stats for the next step
                running_loss = 0
                for key in self._loss_weights.keys():
                    running_loss_dict[key] = 0.0
                t0 = time.perf_counter()

            self.epochs_run += 1
            # save checkpoints within a set interval
            self.save_checkpoint(epoch=curr_epoch)

            if self._do_validation:
                self.validate()
                self._model.train()
                if self._discriminator is not None:
                    self._discriminator.train()

    @torch.inference_mode()
    def validate(self) -> None:
        self._autoencoder.eval()
        self._model.eval()

        max_steps = self._validation_max_steps
        total_steps = len(self._dataloader_validation)
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)

        if len(self._metrics_dict) == 0:
            return

        pbar = tqdm(total=total_steps)
        results_dict = {key: [] for key in self._metrics_dict.keys()}

        for idx, batch in enumerate(self._dataloader_validation):
            if max_steps is not None and idx >= max_steps:
                break

            if self._use_preencoded_latents_validation:
                latents, metadata = batch[0], batch[1]
                latents = latents.to(self._device, dtype=self._dtype)
                latent_targets = latents.clone().detach()
            else:
                waveforms, metadata = batch[0], batch[1]
                waveforms = waveforms.to(self._device, dtype=self._dtype)
                latent_representations, _ = self._autoencoder.encode(waveforms, return_info=True)
                latents = latent_representations
                latent_targets = latent_representations.clone().detach()

            pred_latents, _ = self._model(latents, return_info=True)

            for key, metric in self._metrics_dict.items():
                metric_value = metric(pred_latents, latent_targets)
                if torch.is_tensor(metric_value):
                    metric_value = metric_value.detach().cpu().item()
                results_dict[key].append(float(metric_value))

            pbar.update(1)

        if any(len(values) == 0 for values in results_dict.values()):
            return

        log_dict = {f"val/{key}": sum(values) / len(values) for key, values in results_dict.items()}
        self._metric_logger.log_dict(log_dict, step=self.global_step)

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
