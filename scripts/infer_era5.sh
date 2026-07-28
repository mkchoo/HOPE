#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${HOPE_DATA_DIR:?Set HOPE_DATA_DIR to the ocean/ERA5 Zarr store.}"
: "${HOPE_CKPT:?Set HOPE_CKPT to the final K8 checkpoint.}"

python "${repo_dir}/infer_era5.py" \
  --ckpt "${HOPE_CKPT}" \
  --data_dir "${HOPE_DATA_DIR}" \
  --rollout_days "${ROLLOUT_DAYS:-215}" \
  --start_date "${START_DATE:-202001010000}" \
  --end_date "${END_DATE:-202512310000}" \
  --out_dir "${HOPE_OUTPUT_DIR:-${repo_dir}/outputs/era5}"
