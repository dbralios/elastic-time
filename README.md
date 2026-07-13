# Elastic Time: Dynamic Frame Rate Bottlenecks for Neural Audio Coding

This repository contains the source code for [Elastic Time: Dynamic Frame Rate Bottlenecks for Neural Audio Coding](https://arxiv.org/abs/2606.27320) (Interspeech 2026).

The release covers the following methods reported in the paper:

- Elastic Time
- Conv-Downsample
- H-Net
- CodecSlime

## Environment

Requirements:

- Python 3.10+
- Linux recommended
- CUDA-capable GPU recommended

First install PyTorch and torchaudio matching your CUDA setup, then install this package:

```bash
python -m pip install -e ".[dev]"
```

Launching training:

```bash
python -m recipes.train_rebottleneck_elastic_time --config path/to/config.yaml
```

Note: This assumes your conda environment has this repository installed in editable mode from the current checkout. 

## Autoencoder Checkpoint

All models are implemented as Re-Bottlenecks of the pretrained Stable Audio Open 1.0 autoencoder. Download the official checkpoint:

- [Stable Audio Open 1.0 `model.ckpt`](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/model.ckpt)

Access may require accepting the model terms on Hugging Face. The model weights remain governed by Stability AI's terms and are not redistributed by this repository.

Set `autoencoder_checkpoint_path` accordingly in the training config files.

## Repository Layout

```text
elastic-time/
|--> rebots/                              Model, dataset, config, and utility code
|--> recipes/                             Training entrypoints
|    `--> configs/rebottleneck/
|         |--> preencoded/                Preencoded-latent training configs
|         `--> raw_audio/                 Raw-audio training configs
|--> external/                            Third-party code
```

### Code Overview

- `rebots/config/`: config parsing and `_component_` instantiation
- `rebots/data/datasets.py`: preencoded latent dataset loader
- `rebots/models/rebottlenecks.py`: ReBottleneck, Elastic Time, and H-Net model families
- `rebots/models/components/`: ConvNeXt stacks, conditioning modules, maskers, and replacement logic
- `rebots/models/components/elastic_time_masker.py`: Elastic Time masking-strategy sampling and offset-mask construction
- `rebots/models/components/elastic_time_replace.py`: greedy/DP frame selection, predictor rollouts, and latent-frame replacement
- `rebots/models/components/codecslime_masker.py`: CodecSlime masking-strategy sampling
- `rebots/models/components/codecslime_replace.py`: CodecSlime segment selection and latent replacement
- `rebots/models/predictors.py`: predictor modules used by Elastic Time
- `rebots/models/discriminators.py`: latent discriminator
- `rebots/modules/`: bottleneck, GAN loss, optimizer, and low-level ConvNeXt blocks
- `recipes/`: training entrypoints and configs


## External Code

This repository contains a small amount of external code under `external/`.

- `external/stable_audio_tools/`: Stable Audio Open components used; see `external/stable_audio_tools/LICENSE`
- `external/auraloss/`: loss implementations used; see `external/auraloss/LICENSE`

The repository setup and parts of the recipe, configuration, and utility scaffolding are based on [torchtune](https://github.com/meta-pytorch/torchtune), which is distributed under the BSD 3-Clause License. Adapted files retain their original copyright notices.


## Data Preparation

The repository supports two training modes.

### Preencoded Latents

Preencoded training reads latent arrays with `rebots.data.datasets.PreEncodedDataset`. This avoids running the pretrained audio encoder inside every training step.

### Raw Audio

Raw-audio configs use `external.stable_audio_tools.src.data.dataset.SampleDataset` and encode waveforms during training. Each `LocalDatasetConfig.path` must point to a directory containing the corresponding audio dataset.

## Method Map

| Paper name | Training entrypoint | Notes |
| --- | --- | --- |
| Elastic Time | `recipes/train_rebottleneck_elastic_time.py` | greedy and DP variants |
| Conv-Downsample | `recipes/train_rebottleneck.py` | fixed-rate baseline |
| H-Net | `recipes/train_rebottleneck_hnet.py` | utilization-controlled H-Net model |
| CodecSlime | `recipes/train_rebottleneck_codecslime_predictor.py`, `recipes/train_rebottleneck_codecslime.py` | CodecSlime two-stage training |

## Before Launching Runs

All release configs use the following path variables:

- `data_root: /path/to/data`
- `output_root: /path/to/experiments`
- `autoencoder_checkpoint_path`

Dataset paths and output directories are derived from these variables. For raw-audio training, also provide the Stable Audio Open checkpoint.

Most documented configs use `WandBLogger`. If you keep that logger, set `WANDB_API_KEY` in your environment.

To log locally without Weights & Biases, replace the `metric_logger` block with `DiskLogger`:

```yaml
metric_logger:
  _component_: rebots.utils.metric_logging.DiskLogger
  log_dir: ${output_dir}/logs
  filename: metrics.txt
```

### Conv-Downsample

```bash
python -m recipes.train_rebottleneck --config recipes/configs/rebottleneck/preencoded/baseline_ds=2.yaml
```

### Elastic Time

Greedy:

```bash
python -m recipes.train_rebottleneck_elastic_time --config recipes/configs/rebottleneck/preencoded/elastic_time_greedy.yaml
```

DP:

```bash
python -m recipes.train_rebottleneck_elastic_time --config recipes/configs/rebottleneck/preencoded/elastic_time_dp.yaml
```

DP widerange:

```bash
python -m recipes.train_rebottleneck_elastic_time --config recipes/configs/rebottleneck/preencoded/elastic_time_dp_widerange.yaml
```

Single-point 0.5 greedy:

```bash
python -m recipes.train_rebottleneck_elastic_time --config recipes/configs/rebottleneck/preencoded/elastic_time_greedy_ds=2.yaml
```

Single-point 0.5 DP:

```bash
python -m recipes.train_rebottleneck_elastic_time --config recipes/configs/rebottleneck/preencoded/elastic_time_dp_ds=2.yaml
```

### H-Net

Range-conditioned H-Net-YOTO:

```bash
python -m recipes.train_rebottleneck_hnet --config recipes/configs/rebottleneck/preencoded/hnet_yoto.yaml
```

Single-point 0.5 H-Net:

```bash
python -m recipes.train_rebottleneck_hnet --config recipes/configs/rebottleneck/preencoded/hnet_ds=2.yaml
```

### CodecSlime

Stage 1, melt predictor:

```bash
python -m recipes.train_rebottleneck_codecslime_predictor --config recipes/configs/rebottleneck/preencoded/codecslime_melt_predictor.yaml
```

Stage 2, final CodecSlime run:

```bash
python -m recipes.train_rebottleneck_codecslime --config recipes/configs/rebottleneck/preencoded/codecslime_cool.yaml
```

`codecslime_cool.yaml` depends on the checkpoint produced by the melt-predictor stage.

## Raw-Audio Training Runs

The configs in `recipes/configs/rebottleneck/raw_audio/` run the same model families as above using on-the-fly encoding with a pretrained Stable Audio Open autoencoder.

## Citation

If you use this code, please cite:

```bibtex
@article{bralios2026elastictime,
  title={Elastic Time: Dynamic Frame Rate Bottlenecks for Neural Audio Coding},
  author={Bralios, Dimitrios and Smaragdis, Paris and Kim, Minje},
  journal={arXiv preprint arXiv:2606.27320},
  year={2026}
}
```

## License

The first-party code in this repository is released under the [MIT License](LICENSE). Thrid-party code and pretrained model weights remain subject to their respective license terms.
