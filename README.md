# FireTrend

FireTrend is a physics-guided spatial-temporal framework for wildfire risk-level forecasting. This code has been revised to match the NeurIPS 2026 manuscript version: it first learns label-free physics-aware representations from historical wildfire risk-score maps and environmental covariates, then trains a downstream classifier for ordinal wildfire risk levels.

## What Changed in This Version

- The prediction target is now an ordinal risk-level map with three classes: `0=low`, `1=mid`, `2=high`.
- Training follows the paper's two-stage design:
  - Stage 1: label-free representation pretraining with temporal, spatial, and cross-modal contrastive losses plus PyroCast latent propagation.
  - Stage 2: downstream risk-level classifier training with weighted cross entropy.
- PyroCast is applied in latent feature space, not as a post-processing correction on the predicted output map.
- The model outputs class logits and probabilities `[B, 3, H, W]`.
- The Oregon subset mentioned in the manuscript is not wired in this release, per project scope.

## Repository Layout

```text
FireTrend/
├── main.py
├── config.yaml
├── modules/
│   ├── firetrend_model.py
│   ├── encoder_spatiotemporal.py
│   ├── contrastive_learning.py
│   ├── pyrocast_physics.py
│   └── losses.py
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   └── seed_utils.py
├── scripts/
│   ├── train_firetrend.sh
│   └── eval_firetrend.sh
└── data_v2/
    ├── CA_wildfire_grid_ERA5_LANDFIRE_aligned.h5
    └── FL_wildfire_grid_ERA5_LANDFIRE_aligned.h5
```

## Data

The loader expects one daily HDF5 file per region under `data_v2/`.

Required keys:

- `wildfire_risk`
- Meteorology: `u10`, `v10`, `d2m`, `t2m`, `msl`, `sp`, `stl1`, `swvl1`
- LANDFIRE/geospatial/fuel: `EVH`, `EVC`, `EVT`, `Aspect`, `Slope`, `Elevation`, `CBD`, `FVH`, `FVC`, `FVT`

The model uses normalized tensors for representation learning. For PyroCast, the dataloader also returns raw `u10/v10` wind drivers so wind direction is preserved after input normalization. `t2m` is used as temperature and `d2m` as the available humidity/moisture proxy.

Risk levels are generated from the continuous provider-defined score in `[0, 100]` using `data.risk_thresholds` in `config.yaml`:

```yaml
risk_thresholds: [33.3333, 66.6667]
```

## Model Pipeline

1. `SpatialTemporalEncoder` fuses wildfire score history, meteorological variables, and geospatial covariates into latent maps `Z`.
2. `MultiViewContrastive` computes:
   - `L_temp`: adjacent-time alignment for the same grid cell.
   - `L_spat`: neighborhood-aware spatial contrast.
   - `L_cross`: cross-modal alignment across score, meteorology, and geospatial branches.
3. `PyroCastPhysics` builds wind-conditioned directional kernels from `u10`, `v10`, temperature, and humidity proxy, then propagates latent states channel-wise.
4. The downstream classifier maps the latest physics-aware representation to three risk-level logits.

The implemented objective is:

```text
L_pretrain = L_cross + lambda_s * L_spat + lambda_t * L_temp + lambda_p * L_pyro
L_cls      = weighted_cross_entropy(logits, risk_level_labels)
```

## Training

Install dependencies in your GPU environment, then run:

```bash
cd FireTrend
python main.py --config config.yaml --train --stage pretrain_then_finetune --region california
```

Useful alternatives:

```bash
python main.py --config config.yaml --train --stage pretrain --region california
python main.py --config config.yaml --train --stage finetune --region california --checkpoint outputs/checkpoints/california_pretrain_best.pth
python main.py --config config.yaml --train --stage joint --region florida
```

The helper script runs the default two-stage workflow:

```bash
bash scripts/train_firetrend.sh
```

Checkpoints are saved under `outputs/checkpoints/`.

## Evaluation

```bash
python main.py \
  --config config.yaml \
  --test \
  --region california \
  --checkpoint outputs/checkpoints/california_finetune_best.pth
```

The evaluator reports macro IoU, macro F1, macro AUPRC, accuracy, and per-class IoU over the ordinal risk labels.

## Configuration Notes

Key method controls live in `config.yaml`:

- `contrastive.lambda_cross`, `contrastive.lambda_spatial`, `contrastive.lambda_temporal`
- `model.num_layers` for stacked spatial-temporal Transformer blocks
- `pyrocast.kernel_size`, `pyrocast.rho`, `pyrocast.sigma_parallel`, `pyrocast.sigma_perp`, `pyrocast.lambda_pyro`
- `training.stage`, `training.pretrain_epochs`, `training.finetune_epochs`
- `training.freeze_encoder_during_finetune`
- `data.risk_thresholds`

For multi-GPU servers, keep `training.data_parallel: true`; PyTorch `DataParallel` is enabled automatically when multiple CUDA devices are visible.

## Citation

```bibtex
@inproceedings{firetrend2026,
  title={FireTrend: Learning Physics-Guided Latent Fire Dynamics for Wildfire Risk-Level Forecasting},
  author={Anonymous},
  booktitle={Submission to NeurIPS},
  year={2026}
}
```
