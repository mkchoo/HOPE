"""Autoregressive HOPE forecast driven by one SEAS5 ensemble member."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import ATM_VARS, DEFAULT_DATA_DIR, InitialConditionDataset
from inference_utils import load_ema_model, parse_time, rollout, save_netcdf, to_physical


PROJECT_DIR = Path(__file__).resolve().parent
ACCUMULATED_VARS = frozenset(("ssr", "strd", "tp"))


class SEAS5Forcing:
    """Read ``ROOT/MEMBER/YYYYMMDDHHMM.nc`` and normalize its daily fields."""

    def __init__(self, root: str, member: str, dataset: InitialConditionDataset):
        self.directory = Path(root) / str(member)
        self.dataset = dataset
        if not self.directory.is_dir():
            raise FileNotFoundError(f"SEAS5 member directory not found: {self.directory}")

    def path(self, init_time: str) -> Path:
        return self.directory / f"{init_time}.nc"

    @staticmethod
    def _dimension(da: xr.DataArray, candidates) -> str:
        for name in candidates:
            if name in da.dims:
                return name
        raise ValueError(f"Cannot find one of {candidates} in dimensions {da.dims}")

    def available_steps(self, init_time: str) -> int:
        path = self.path(init_time)
        if not path.is_file():
            return 0
        with xr.open_dataset(path) as ds:
            first = ds[ATM_VARS[0]]
            time_dim = self._dimension(first, ("time", "lead_time", "step"))
            return int(first.sizes[time_dim])

    def load(self, init_time: str, steps: int) -> np.ndarray:
        path = self.path(init_time)
        with xr.open_dataset(path) as ds:
            missing = [name for name in ATM_VARS if name not in ds]
            if missing:
                raise KeyError(f"{path} is missing {missing}")

            fields = []
            for name in ATM_VARS:
                da = ds[name]
                time_dim = self._dimension(da, ("time", "lead_time", "step"))
                lat_dim = self._dimension(da, ("lat", "latitude", "y"))
                lon_dim = self._dimension(da, ("lon", "longitude", "x"))
                da = da.transpose(time_dim, lat_dim, lon_dim).isel({time_dim: slice(0, steps)})

                if lat_dim in da.coords:
                    source_lat = np.asarray(da[lat_dim].values)
                    if np.allclose(source_lat[::-1], self.dataset.lat_values, atol=1e-4):
                        da = da.isel({lat_dim: slice(None, None, -1)})
                    elif not np.allclose(source_lat, self.dataset.lat_values, atol=1e-4):
                        raise ValueError(f"{path}: SEAS5 latitude grid does not match the ocean grid")
                if lon_dim in da.coords and not np.allclose(
                    np.asarray(da[lon_dim].values), self.dataset.lon_values, atol=1e-4
                ):
                    raise ValueError(f"{path}: SEAS5 longitude grid does not match the ocean grid")

                array = np.asarray(da.values, dtype=np.float32)
                if name in ACCUMULATED_VARS:
                    array = array / 24.0
                normalized = self.dataset.normalizer.normalize(
                    name, array[:, None], atmosphere=True
                )
                fields.append(normalized)

        return np.concatenate(fields, axis=1).astype(np.float32)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HOPE SEAS5-forced rollout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", default=os.environ.get("HOPE_CKPT"))
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--seas5_root", default=os.environ.get("SEAS5_ROOT"))
    parser.add_argument("--member", default="1")
    parser.add_argument("--start_date", default="202001010000")
    parser.add_argument("--end_date", default="202508010000")
    parser.add_argument("--rollout_days", type=int, default=215)
    parser.add_argument("--warmup_days", type=int, default=2)
    parser.add_argument("--out_dir", default=str(PROJECT_DIR / "outputs/seas5"))
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def select_initializations(
    dataset: InitialConditionDataset,
    forcing: SEAS5Forcing,
    start: str,
    end: str,
) -> None:
    start_time = parse_time(start)
    end_time = parse_time(end)
    selected = []
    for index in dataset.valid_start_indices:
        init_time = dataset.file_dates[index]
        lead1 = parse_time(dataset.file_dates[index + 1])
        leadk = parse_time(dataset.file_dates[index + dataset.rollout_days])
        if lead1 < start_time or leadk > end_time:
            continue
        if forcing.available_steps(init_time) < 1 and dataset.rollout_days > 1:
            continue
        selected.append(index)
    if not selected:
        raise ValueError("No initializations have matching SEAS5 files in the requested period")
    dataset.valid_start_indices = selected


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if not args.ckpt:
        raise ValueError("Set --ckpt or HOPE_CKPT")
    if not args.seas5_root:
        raise ValueError("Set --seas5_root or SEAS5_ROOT")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.out_dir) / str(args.member)
    output.mkdir(parents=True, exist_ok=True)
    dataset = InitialConditionDataset(
        args.data_dir,
        args.rollout_days,
        args.warmup_days,
        cache_days=args.warmup_days + 1,
    )
    seas5 = SEAS5Forcing(args.seas5_root, args.member, dataset)
    select_initializations(dataset, seas5, args.start_date, args.end_date)
    model = load_ema_model(args.ckpt, dataset, device)
    valid_mask = dataset.valid_mask_tensor(device)

    loader_options = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
    }
    if args.num_workers:
        loader_options.update(persistent_workers=True, prefetch_factor=1)
    data_loader = DataLoader(dataset, **loader_options)

    with (output / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": args.ckpt,
                "checkpoint_state": "ema_model",
                "forcing": "SEAS5",
                "member": str(args.member),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "requested_rollout_days": args.rollout_days,
                "warmup_days": args.warmup_days,
                "first_SEAS5_field": "init + 24 hours; used after the first ocean prediction",
                "accumulated_fields": "ssr, strd, tp divided by 24 before normalization",
            },
            handle,
            indent=2,
        )

    saved = skipped = shortened = 0
    for batch in tqdm(data_loader, desc=f"SEAS5 member {args.member} rollout"):
        start_index = int(batch["start_index"].item())
        init_time = dataset.file_dates[start_index]
        path = output / f"{init_time}.nc"
        if path.exists() and not args.overwrite:
            skipped += 1
            continue

        available = seas5.available_steps(init_time)
        forecast_days = min(args.rollout_days, available + 1)
        forcing = torch.from_numpy(seas5.load(init_time, forecast_days - 1))[None]
        prediction = rollout(
            model,
            batch["input"],
            batch["warmup_inputs"],
            batch["aux_future"][:, : forecast_days - 1],
            forcing,
            valid_mask,
        )[0].float().cpu().numpy()
        save_netcdf(
            path,
            dataset,
            to_physical(dataset, prediction),
            init_time=init_time,
        )
        saved += 1
        shortened += int(forecast_days < args.rollout_days)

    print(
        f"SEAS5 member {args.member}: saved={saved}, shortened={shortened}, "
        f"skipped_existing={skipped}, output={output}"
    )


if __name__ == "__main__":
    main()
