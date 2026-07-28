#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${HOPE_DATA_DIR:?Set HOPE_DATA_DIR to the ocean/ERA5 Zarr store.}"
out_root="${HOPE_OUT_ROOT:-${repo_dir}/runs}"

accelerate launch "${repo_dir}/train_rollout_mask.py" \
  --stage K1 \
  --data_dir "${data_dir}" \
  --out_root "${out_root}"

k1_ckpt="${out_root}/hope_K1/checkpoints/epoch_0250.pt"
accelerate launch "${repo_dir}/train_rollout_mask.py" \
  --stage K4 \
  --data_dir "${data_dir}" \
  --out_root "${out_root}" \
  --init_ckpt "${k1_ckpt}"

k4_ckpt="${out_root}/hope_K4/checkpoints/epoch_0100.pt"
accelerate launch "${repo_dir}/train_rollout_mask.py" \
  --stage K8 \
  --data_dir "${data_dir}" \
  --out_root "${out_root}" \
  --init_ckpt "${k4_ckpt}"
