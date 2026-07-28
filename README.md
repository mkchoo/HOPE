# HOPE ocean–cryosphere forecasting

Minimal training and inference code for:

> A hierarchical neural emulator for stable long-term prediction of the
> atmosphere-driven ocean–cryosphere system

Participation-ratio analysis, latent diagnostics, plotting, alternative
normalizations, experimental losses, and unused model modes are not included.

![HOPE model flow](assets/hope_ocean_cryosphere_flow.png)

## What remains

```text
dataloader.py              Fixed ocean/ERA5 Zarr pipeline
train_rollout_mask.py      K1/K4/K8 EMA curriculum training
infer_era5.py              ERA5-forced autoregressive hindcast
infer_seas5.py             Single-member SEAS5 forecast
inference_utils.py         Shared rollout and NetCDF writer
models/                    HOPE, LightUNet++, and longitude-shifted Swin
scripts/                   One-command examples
logs/                      Original K1/K4/K8 training records
data/                      Training normalization table
```

The model has one released configuration:

- state: `siconc`, `sithick`, `zos`, `thetao`, `so`, `uo`, `vo` (95 channels);
- auxiliary: land mask and solar-zenith cosine (2 channels);
- atmosphere: `t2m`, `d2m`, `msl`, `u10`, `v10`, `ssr`, `strd`, `tp` (8 channels);
- input/output: 105/95 channels on a `180 × 360` grid;
- two low-level updates per cycle and two high-level cycles;
- latitude positional encoding, recurrent `zL/zH`, and a residual state head.

`usi` and `vsi` are not part of this experiment.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
accelerate config
```

The ocean/ERA5 data must be a daily, consolidated Zarr store with `time`,
`lev`, `lat`, and `lon` coordinates and all variables listed above. The full
dataset and model checkpoints are too large to include.

## Train

The curriculum settings are fixed in the code because these are the runs used
for the released model:

| Stage | Rollout | Warm-up | Epochs | LR | Batch |
|---|---:|---:|---:|---:|---:|
| K1 | 1 | 0 | 250 | `2e-4` | 6 |
| K4 | 4 | 1 | 100 | `1e-4` | 2 |
| K8 | 8 | 2 | 100 | `1e-4` | 1 |

K4 loads the K1 EMA weights, and K8 loads the K4 EMA weights. All stages use
BF16, Adam-atan2, EMA 0.999, gradient clipping 1.0, ocean validity masks, and
the same variable/latitude-weighted MAE.

```bash
export HOPE_DATA_DIR=/path/to/dataset_final.zarr
bash scripts/train_curriculum.sh
```

Run one stage:

```bash
accelerate launch train_rollout_mask.py \
  --stage K8 \
  --data_dir /path/to/dataset_final.zarr \
  --init_ckpt /path/to/K4/epoch_0100.pt
```

Resume an interrupted stage:

```bash
accelerate launch train_rollout_mask.py \
  --stage K8 \
  --data_dir /path/to/dataset_final.zarr \
  --resume_ckpt /path/to/K8/epoch_0050.pt
```

## ERA5 inference

ERA5 inference feeds the predicted ocean–cryosphere state back into the model.
Future atmosphere and auxiliary fields come from the Zarr store.

```bash
export HOPE_DATA_DIR=/path/to/dataset_final.zarr
export HOPE_CKPT=/path/to/epoch_0100.pt
bash scripts/infer_era5.sh
```

Only month-start initializations are selected by default. Use
`--start_day 0` to forecast every valid day.

## SEAS5 inference

SEAS5 files use this layout:

```text
SEAS5_ROOT/
└── MEMBER/
    └── YYYYMMDDHHMM.nc
```

Each file contains the eight atmospheric variables with a daily time/lead
dimension. The first field is `initialization + 24 h`. Warm-up and the initial
input atmosphere come from ERA5; subsequent rollout forcing comes from SEAS5.
The accumulated `ssr`, `strd`, and `tp` fields are divided by 24 before applying
the ERA5 normalization.

```bash
export HOPE_DATA_DIR=/path/to/dataset_final.zarr
export HOPE_CKPT=/path/to/epoch_0100.pt
export SEAS5_ROOT=/path/to/seas5
export SEAS5_MEMBER=1
bash scripts/infer_seas5.sh
```

Run the script once per ensemble member. Forecasts shorter than the requested
horizon are saved with the available SEAS5 lead length.

## Original training logs

`logs/` preserves the original K1, K4, and K8 arguments, TensorBoard event
files, scalar CSVs, and `training_summary.csv`.

```bash
tensorboard --logdir logs
```

The final model was optimized on eight-day rollouts. A 215-day run is therefore
autoregressive extrapolation beyond the training horizon.
