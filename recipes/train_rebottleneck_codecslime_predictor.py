import inspect
import sys
import time
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rebots import config, utils
from rebots.models.components.codecslime_replace import mix_with_codecslime_segments
from rebots.models.components.elastic_time_replace import segment_matrix_to_segment_length_histogram
from rebots.utils.lr_schedulers import get_lr
from recipes.train_rebottleneck_codecslime import TrainingRecipeSingleDevice as CodecSlimeTrainingRecipeBase

log = utils._logging.get_logger("DEBUG")


class TrainingRecipeSingleDevice(CodecSlimeTrainingRecipeBase):
    """
    Codec-Slime recipe with an auxiliary Elastic Time predictor objective.

    The predictor is trained only through the Predictor loss. Its outputs are not
    used for latent placement/mixing in the reconstruction path.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._predictor_accepts_state = False

        self._K = int(cfg.get("K", 12))
        self._K_rollout = int(cfg.get("K_rollout", self._K))
        if self._K < 1:
            raise ValueError(f"K must be >= 1, got {self._K}")
        if self._K_rollout < 1:
            raise ValueError(f"K_rollout must be >= 1, got {self._K_rollout}")

    def setup(self, cfg: DictConfig) -> None:
        super().setup(cfg)

        if not hasattr(self._model, "predictor"):
            raise ValueError("This recipe expects model.predictor (e.g., ElasticTimeReBottleneck).")

        self._predictor_forward = self._maybe_compile_callable(self._model.predictor, "model.predictor")
        self._predictor_params = [p for p in self._model.predictor.parameters() if p.requires_grad]

        try:
            predictor_signature = inspect.signature(self._model.predictor.forward)
            self._predictor_accepts_state = len(predictor_signature.parameters) >= 2
        except (TypeError, ValueError):
            self._predictor_accepts_state = False

        if "loss_predictor" in cfg:
            self._loss_fn_predictor = config.instantiate(cfg.loss_predictor)
        else:
            self._loss_fn_predictor = config.instantiate(cfg.loss)

        predictor_params = sum(p.numel() for p in self._model.predictor.parameters())
        log.info(f"Predictor parameters: {predictor_params:,}")

    def _predictor_step(
        self,
        z: torch.Tensor,
        h_state: Optional[torch.Tensor],
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

    def train(self) -> None:
        if self._compile:
            target_modules = "model encode/decode/predictor"
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

                loss_predictor = torch.zeros((), device=self._device, dtype=z.dtype)
                predictor_error_by_k: list[float] = []
                if (
                    self._loss_weights.get("loss_predictor", None) is not None
                    and self._K_rollout > 0
                    and z.shape[-1] > 1
                ):
                    max_rollout = min(self._K_rollout, self._K, z.shape[-1] - 1)
                    current = z.detach()
                    h_state = None
                    total_valid = 0

                    for k in range(1, max_rollout + 1):
                        current, h_state = self._predictor_step(current, h_state)
                        pred_k = current[:, :, : z.shape[-1] - k]
                        targ_k = z.detach()[:, :, k:]

                        residual = pred_k - targ_k
                        scale = targ_k.std(unbiased=False).clamp_min(1e-5)
                        residual_normalized = residual / scale

                        num_k = residual.pow(2).mean().detach()
                        den_k = targ_k.pow(2).mean().detach().clamp_min(1e-8)
                        predictor_error_by_k.append(float((num_k / den_k).item()))

                        loss_predictor = loss_predictor + (z.shape[-1] - k) * self._loss_fn_predictor(
                            residual_normalized,
                            torch.zeros_like(residual_normalized),
                        )
                        total_valid += z.shape[-1] - k

                    if total_valid > 0:
                        loss_predictor = loss_predictor / total_valid

                h = self._masker.replace(z.detach(), mask_sample)
                z_dec = mix_with_codecslime_segments(z, h)

                if self._conditional_decoder:
                    pred_latents = self._model_decode(z_dec, cond_input=cond_input)
                else:
                    pred_latents = self._model_decode(z_dec)

                loss: dict[str, Any] = {}
                loss_d: dict[str, Any] = {}

                loss["loss_reconstruction"] = self._loss_fn(pred_latents, latent_targets)
                loss["loss_predictor"] = loss_predictor
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
                    if len(predictor_error_by_k) > 0:
                        log_dict["stats/predictor_nmse_mean"] = sum(predictor_error_by_k) / len(predictor_error_by_k)

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

                        if len(predictor_error_by_k) > 0:
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
