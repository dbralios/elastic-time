import inspect
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional
from warnings import warn

import torch
import torch._functorch.config as functorch_config
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from rebots import config, utils
from rebots.data.datasets import PreEncodedDataset
from rebots.models.components.elastic_time_replace import (
    mix_with_elastic_time_segments,
    segment_matrix_to_segment_length_histogram,
)
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
        self._wandb_hist_every_n_steps = cfg.get("wandb_hist_every_n_steps", 100)

        # Training cfg
        self._gradient_accumulation_steps = 1
        if cfg.get("gradient_accumulation_steps", 1) != 1:
            raise NotImplementedError("Gradient accumulation is not supported for this recipe. Please set it to 1.")
        self._resume_from_checkpoint = cfg.get("resume_from_checkpoint", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._conditional_encoder = cfg.get("conditional_encoder", False)
        self._conditional_decoder = cfg.get("conditional_decoder", False)
        self._predictor_accepts_state = False

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = cfg.get("seed", None)
        utils.set_seed(seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None))
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self.save_every_n_epochs = cfg.get("save_every_n_epochs", 3)

        self._do_validation = False
        self._validation_max_steps = None
        self._use_preencoded_latents_train = False
        self._use_preencoded_latents_validation = False
        self._masker_validation = None
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
        """Updates the recipe state from checkpoint."""
        # Make sure keys are present
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

        # warn the user but *allow* the override
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
        self._compile_disable_donated_buffer = cfg.get("compile_disable_donated_buffer", True)

        if self._compile and self._compile_disable_donated_buffer:
            functorch_config.donated_buffer = False
            log.info("torch.compile: set torch._functorch.config.donated_buffer=False")

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

        self._model_encode, self._model_decode, self._predictor_forward = self._setup_model_ops(self._model)
        self._discriminator_forward = self._setup_discriminator_ops(self._discriminator)
        self._predictor_params = [p for p in self._model.predictor.parameters() if p.requires_grad]
        self._optimizer_predictor = None
        self._lr_scheduler_predictor = None
        self._has_separate_predictor_optimizer = cfg.get("optimizer_predictor", None) is not None

        predictor_param_ids = {id(p) for p in self._predictor_params}
        self._main_model_params = [
            p
            for p in self._model.parameters()
            if p.requires_grad and (not self._has_separate_predictor_optimizer or id(p) not in predictor_param_ids)
        ]

        autoencoder_params = sum(p.numel() for p in self._autoencoder.parameters())
        model_params = sum(p.numel() for p in self._model.parameters())
        predictor_params = sum(p.numel() for p in self._model.predictor.parameters())
        log.info(f"Autoencoder parameters: {autoencoder_params:,}")
        log.info(f"Model parameters: {model_params:,}")
        log.info(f"Predictor parameters: {predictor_params:,}")
        if self._discriminator is not None:
            discriminator_params = sum(p.numel() for p in self._discriminator.parameters())
            log.info(f"Discriminator parameters: {discriminator_params:,}")

        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(ckpt_dict[utils.OPT_KEY] if ckpt_dict and utils.OPT_KEY in ckpt_dict else None),
            params=self._main_model_params,
        )

        if self._has_separate_predictor_optimizer:
            self._optimizer_predictor = self._setup_optimizer(
                cfg_optimizer=cfg.optimizer_predictor,
                params=self._predictor_params,
                opt_state_dict=(
                    ckpt_dict[utils.OPT_PREDICTOR_KEY] if ckpt_dict and utils.OPT_PREDICTOR_KEY in ckpt_dict else None
                ),
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
        if "loss_predictor" in cfg:
            self._loss_fn_predictor = config.instantiate(cfg.loss_predictor)
        else:
            self._loss_fn_predictor = config.instantiate(cfg.loss)

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
            # Warn the user if any recipe state was not loaded from checkpoint
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

        if self._optimizer_predictor is not None:
            self._lr_scheduler_predictor = self._setup_lr_scheduler(
                optimizer=self._optimizer_predictor,
                cfg_lr_scheduler=cfg.get("lr_scheduler_predictor", None),
                num_training_steps=self.total_epochs * self._steps_per_epoch,
                last_epoch=self.global_step - 1,
            )

        # Setup loss weights
        self._loss_weights = cfg.get("loss_weights", None)

        # Setup masker
        self._K = cfg.get("K", 16)
        self._K_rollout = cfg.get("K_rollout", self._K)
        self._masker = config.instantiate(cfg.masker, K=self._K)
        if self._do_validation:
            cfg_masker_validation = cfg.get("masker_validation", None)
            if cfg_masker_validation is None:
                raise ValueError("dataset_validation is set, but no `masker_validation` was configured in OmegaConf.")
            self._masker_validation = config.instantiate(cfg_masker_validation, K=self._K)

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

    def _setup_model_ops(self, model: nn.Module) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
        encode_fn = self._maybe_compile_callable(model.encode, "model.encode")
        decode_fn = self._maybe_compile_callable(model.decode, "model.decode")
        predictor_fn = self._maybe_compile_callable(model.predictor, "model.predictor")

        try:
            predictor_signature = inspect.signature(model.predictor.forward)
            self._predictor_accepts_state = len(predictor_signature.parameters) >= 2
        except (TypeError, ValueError):
            self._predictor_accepts_state = False

        return encode_fn, decode_fn, predictor_fn

    def _predictor_step(
        self, z: torch.Tensor, h_state: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._predictor_accepts_state:
            pred = self._predictor_forward(z, h_state) if h_state is not None else self._predictor_forward(z)
        else:
            pred = self._predictor_forward(z)

        if isinstance(pred, tuple):
            if len(pred) != 2:
                raise ValueError(f"Expected predictor output tuple of length 2, got {len(pred)}")
            z_next, h_next = pred
            return z_next, h_next

        return pred, None

    def _predictor_output_only(self, z: torch.Tensor) -> torch.Tensor:
        z_next, _ = self._predictor_step(z, h_state=None)
        return z_next

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
            try:
                optimizer.load_state_dict(opt_state_dict)
            except ValueError as e:
                log.error(f"Error loading optimizer state dict: {e}")

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
            if self._optimizer_predictor is not None:
                ckpt_dict[utils.OPT_PREDICTOR_KEY] = self._optimizer_predictor.state_dict()

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
        if self._compile:
            target_modules = "model encode/decode/predictor"
            if self._discriminator is not None and self._compile_discriminator:
                target_modules += " and discriminator"
            log.info(f"NOTE: torch.compile is enabled for {target_modules}. Expect a relatively slow first iteration.")

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
        if self._optimizer_predictor is not None:
            self._optimizer_predictor.zero_grad()

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
                    autoencoder_inputs = waveforms  # torch.cat([waveforms, targets], dim=0)

                    # Encode the waveforms to latent representations
                    with torch.no_grad():
                        latent_representations, encoder_info = self._autoencoder.encode(
                            autoencoder_inputs, return_info=True
                        )
                    # Split the latent representations into two parts
                    # latents, latent_targets = torch.chunk(latent_representations, 2, dim=0)
                    latents = latent_representations
                    latent_targets = latent_representations.clone().detach()

                # Sample masking plan and feed the latents to the model
                total_training_steps = max(1, self.total_epochs * self._steps_per_epoch)
                if total_training_steps <= 1:
                    masker_progress = 1.0
                else:
                    masker_progress = float(self.global_step) / float(total_training_steps - 1)
                mask_sample = self._masker.sample(
                    latents.shape[0], latents.shape[-1], latents.device, progress=masker_progress
                )
                cond_input = mask_sample.target_replace_fraction
                if self._conditional_encoder:
                    z, info = self._model_encode(latents, return_info=True, cond_input=cond_input)
                else:
                    z, info = self._model_encode(latents, return_info=True)

                # Elastic Time predictions
                B, C, T = z.shape
                max_K = self._K
                device, dtype = z.device, z.dtype

                current = z
                h_state = None

                # (B, max_K, C, T)
                z_preds = torch.zeros(B, max_K, C, T, device=device, dtype=dtype)
                for k in range(max_K):
                    # current = A^k z_t
                    current, h_state = self._predictor_step(current, h_state)
                    z_preds[:, k, :, :] = current

                loss_predictor = torch.zeros((), device=device, dtype=dtype)
                predictor_error_by_k = []
                if self._loss_weights.get("loss_predictor", None) is not None:
                    total_valid = 0
                    for k in range(1, self._K_rollout + 1):
                        # z_preds index is k-1 because z_preds[:,0] = A^1 z
                        pred_k = z_preds[:, k - 1, :, : T - k]  # (B, C, T - k)
                        targ_k = z[:, :, k:].detach()  # (B, C, T - k)
                        residual = pred_k - targ_k
                        scale = targ_k.std(unbiased=False).clamp_min(1e-5)
                        residual_normalized = residual / scale

                        num_k = residual.pow(2).mean().detach()
                        den_k = targ_k.pow(2).mean().detach().clamp_min(1e-8)
                        predictor_error_by_k.append(float((num_k / den_k).item()))

                        loss_predictor = loss_predictor + (T - k) * self._loss_fn_predictor(
                            residual_normalized, torch.zeros_like(residual_normalized)
                        )
                        total_valid += T - k

                    loss_predictor = loss_predictor / total_valid

                # Sample segments and place them accordingly
                h = self._masker.replace(z, self._model.predictor, mask_sample)
                z_dec = mix_with_elastic_time_segments(z, z_preds, h)

                if self._conditional_decoder:
                    pred_latents = self._model_decode(z_dec, cond_input=cond_input)
                else:
                    pred_latents = self._model_decode(z_dec)

                # Compute loss
                loss = {}
                loss_d = {}

                # Reconstruction loss
                loss["loss_reconstruction"] = self._loss_fn(pred_latents, latent_targets)
                # Predictor loss
                loss["loss_predictor"] = loss_predictor
                # Predictor loss valid
                valid_replace = h > 0
                if valid_replace.any():
                    b_idx, t_idx = valid_replace.nonzero(as_tuple=True)
                    pred_valid = z_dec[b_idx, :, t_idx]
                    targ_valid = z.detach()[b_idx, :, t_idx]
                    residual_valid = pred_valid - targ_valid
                    scale = targ_valid.std(unbiased=False).clamp(min=1e-6)
                    residual_normalized = residual_valid / scale
                    loss["loss_predictor_valid"] = self._loss_fn_predictor(
                        residual_normalized, torch.zeros_like(residual_normalized)
                    )
                    loss_predictor_valid_var_normalized = (residual_valid**2).mean() / (
                        torch.var(targ_valid, unbiased=False) + 1e-6
                    )
                else:
                    loss["loss_predictor_valid"] = torch.zeros((), device=self._device, dtype=z.dtype)
                    loss_predictor_valid_var_normalized = None

                # KL loss
                loss["loss_kl"] = info.get("kl", torch.zeros(1, device=self._device))
                # VQ losses
                loss["loss_commitment"] = info.get("commitment_loss", torch.zeros(1, device=self._device)).mean()
                loss["loss_codebook"] = info.get("codebook_loss", torch.zeros(1, device=self._device)).mean()

                # Discriminator training
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
                    d_real, fm_real = self._discriminator_forward(latent_targets)
                    d_fake, fm_fake = self._discriminator_forward(pred_latents)

                    loss["loss_adversarial"] = self._loss_gan.generator_loss(d_fake, d_real)
                    loss["loss_feature_matching"] = self._loss_gan.feature_matching_loss(fm_fake, fm_real)

                current_loss = 0.0
                current_loss_main = torch.zeros((), device=self._device, dtype=pred_latents.dtype)
                current_loss_predictor = torch.zeros((), device=self._device, dtype=pred_latents.dtype)
                for key in loss.keys():
                    if self._loss_weights.get(key, None) is not None:
                        running_loss_dict[key] += loss[key].item()
                        weighted_loss = self._loss_weights[key] * loss[key]
                        current_loss += weighted_loss
                        if key in {"loss_predictor", "loss_predictor_valid"}:
                            current_loss_predictor = current_loss_predictor + weighted_loss
                        else:
                            current_loss_main = current_loss_main + weighted_loss

                running_loss += current_loss

                keep_graph = bool(current_loss_predictor.requires_grad)
                if current_loss_main.requires_grad:
                    current_loss_main.backward(retain_graph=keep_graph)

                if current_loss_predictor.requires_grad:
                    if self._predictor_params:
                        current_loss_predictor.backward(inputs=self._predictor_params)
                    else:
                        current_loss_predictor.backward()

                # Step with optimizer
                if self._clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._main_model_params, max_norm=float(self._clip_grad_norm)
                    )
                    if self._optimizer_predictor is not None:
                        grad_norm_predictor = torch.nn.utils.clip_grad_norm_(
                            self._predictor_params,
                            max_norm=float(self._clip_grad_norm),
                        )
                self._optimizer.step()
                if self._optimizer_predictor is not None:
                    self._optimizer_predictor.step()
                self._optimizer.zero_grad(set_to_none=True)
                if self._optimizer_predictor is not None:
                    self._optimizer_predictor.zero_grad(set_to_none=True)

                if self._lr_scheduler is not None:
                    self._lr_scheduler.step()
                if self._lr_scheduler_predictor is not None:
                    self._lr_scheduler_predictor.step()
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
                        "stats/latent_std": torch.std(z).item(),
                    }
                    if self._optimizer_predictor is not None:
                        log_dict["lr_predictor"] = get_lr(self._optimizer_predictor)
                    if loss_predictor_valid_var_normalized is not None:
                        log_dict["stats/loss_predictor_valid_norm"] = loss_predictor_valid_var_normalized.item()

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
                            h.detach().to(dtype=torch.long), K=self._K
                        )
                        bin_rows = [[k, value] for k, value in enumerate(h_bin_avg.tolist())]

                        bin_table = wandb.Table(data=bin_rows, columns=["bin", "avg_count"])
                        log_dict["stats/h_bin_plot"] = wandb.plot.bar(
                            bin_table,
                            "bin",
                            "avg_count",
                            title="Elastic Time Offset Bin Avg Count",
                        )

                        if predictor_error_by_k:
                            error_rows = [[k + 1, value] for k, value in enumerate(predictor_error_by_k)]
                            error_table = wandb.Table(data=error_rows, columns=["k", "nmse"])
                            log_dict["stats/predictor_error_by_k_plot"] = wandb.plot.bar(
                                error_table,
                                "k",
                                "nmse",
                                title="Predictor Error by K",
                            )

                    if self._clip_grad_norm is not None:
                        log_dict.update({"grad_norm": grad_norm})
                        if self._optimizer_predictor is not None:
                            log_dict.update({"grad_norm_predictor": grad_norm_predictor})
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

            # save checkpoints within a set interval
            if self.epochs_run % self.save_every_n_epochs == 0 or (curr_epoch + 1) == self.total_epochs:
                self.save_checkpoint(epoch=curr_epoch)
            self.epochs_run += 1

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
        results_dict = {}
        for key in self._metrics_dict.keys():
            results_dict[f"{key}_base"] = []
            results_dict[f"{key}_masked"] = []

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
                with torch.no_grad():
                    latent_representations, _ = self._autoencoder.encode(waveforms, return_info=True)
                latents = latent_representations
                latent_targets = latent_representations.clone().detach()

            mask_sample = self._masker_validation.sample(latents.shape[0], latents.shape[-1], latents.device)
            cond_input_masked = mask_sample.target_replace_fraction
            cond_input_base = torch.zeros_like(cond_input_masked)

            if self._conditional_encoder:
                z_base, _ = self._model_encode(latents, return_info=True, cond_input=cond_input_base)
                z_masked, _ = self._model_encode(latents, return_info=True, cond_input=cond_input_masked)
            else:
                z_masked, _ = self._model_encode(latents, return_info=True)
                z_base = z_masked

            if self._conditional_decoder:
                pred_latents_base = self._model_decode(z_base, cond_input=cond_input_base)
            else:
                pred_latents_base = self._model_decode(z_base)

            B, C, T = z_masked.shape
            max_K = self._K
            current = z_masked
            h_state = None
            z_preds = torch.zeros(B, max_K, C, T, device=z_masked.device, dtype=z_masked.dtype)
            for k in range(max_K):
                current, h_state = self._predictor_step(current, h_state)
                z_preds[:, k, :, :] = current

            h = self._masker_validation.replace(z_masked, self._model.predictor, mask_sample)
            z_dec = mix_with_elastic_time_segments(z_masked, z_preds, h)

            if self._conditional_decoder:
                pred_latents_masked = self._model_decode(z_dec, cond_input=cond_input_masked)
            else:
                pred_latents_masked = self._model_decode(z_dec)

            for key, metric in self._metrics_dict.items():
                metric_value_base = metric(pred_latents_base, latent_targets)
                if torch.is_tensor(metric_value_base):
                    metric_value_base = metric_value_base.detach().cpu().item()

                metric_value_masked = metric(pred_latents_masked, latent_targets)
                if torch.is_tensor(metric_value_masked):
                    metric_value_masked = metric_value_masked.detach().cpu().item()

                results_dict[f"{key}_base"].append(float(metric_value_base))
                results_dict[f"{key}_masked"].append(float(metric_value_masked))

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
