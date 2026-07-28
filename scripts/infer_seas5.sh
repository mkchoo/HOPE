#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${HOPE_DATA_DIR:?Set HOPE_DATA_DIR to the ocean/ERA5 Zarr store.}"
: "${HOPE_CKPT:?Set HOPE_CKPT to the final K8 checkpoint.}"
: "${SEAS5_ROOT:?Set SEAS5_ROOT to MEMBER/YYYYMMDDHHMM.nc directories.}"

python "${repo_dir}/infer_seas5.py" \
  --ckpt "${HOPE_CKPT}" \
  --data_dir "${HOPE_DATA_DIR}" \
  --seas5_root "${SEAS5_ROOT}" \
  --member "${SEAS5_MEMBER:-1}" \
  --rollout_days "${ROLLOUT_DAYS:-215}" \
  --start_date "${START_DATE:-202001010000}" \
  --end_date "${END_DATE:-202508010000}" \
  --out_dir "${HOPE_OUTPUT_DIR:-${repo_dir}/outputs/seas5}"
