"""Data pipeline used by the released HOPE experiment.

The ocean/ERA5 archive is a single daily, time-major Zarr store.  State,
auxiliary, and atmospheric channel definitions are intentionally fixed here so
the training and inference scripts cannot silently select a different
experiment.
"""

from __future__ import annotations

import datetime as dt
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = os.environ.get("HOPE_DATA_DIR", str(PROJECT_DIR / "data/dataset_final.zarr"))
DEFAULT_NORM_CSV = PROJECT_DIR / "data/stats_final_no_flux_19930101_20171231.csv"

STATE_VARS = ("siconc", "sithick", "zos", "thetao", "so", "uo", "vo")
ATM_VARS = ("t2m", "d2m", "msl", "u10", "v10", "ssr", "strd", "tp")
MINMAX_STATE_VARS = frozenset(("siconc", "sithick", "zos", "thetao", "so"))
SYMMETRIC_STATE_VARS = frozenset(("uo", "vo"))
MINMAX_ATM_VARS = frozenset(("t2m", "d2m", "msl", "ssr", "strd", "tp"))
SYMMETRIC_ATM_VARS = frozenset(("u10", "v10"))


def _time_token(value) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d%H%M")


def _to_datetime(token: str) -> dt.datetime:
    return dt.datetime.strptime(str(token), "%Y%m%d%H%M")


def as_chw(da: xr.DataArray) -> np.ndarray:
    """Convert a 2-D or level-dependent field to ``(channel, lat, lon)``."""
    da = da.squeeze(drop=True)
    if "lev" in da.dims:
        spatial = [d for d in da.dims if d != "lev"]
        if len(spatial) != 2:
            raise ValueError(f"Expected (lev, lat, lon), got {da.dims}")
        da = da.transpose("lev", *spatial)
        return np.asarray(da.values, dtype=np.float32)
    if da.ndim != 2:
        raise ValueError(f"Expected a 2-D field, got {da.dims}")
    return np.asarray(da.values, dtype=np.float32)[None]


class Normalizer:
    """The exact normalization used for the published experiment."""

    def __init__(self, csv_path: str | Path, levels: np.ndarray):
        table = pd.read_csv(csv_path)
        self.levels = np.asarray(levels, dtype=np.float32)
        self.stats: Dict[str, Dict[str, np.ndarray | float | bool]] = {}

        for name, rows in table.groupby("var"):
            if rows["lev"].isna().all():
                row = rows.iloc[0]
                self.stats[str(name)] = {
                    "has_lev": False,
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "min": float(row["min"]),
                    "max": float(row["max"]),
                }
                continue

            rows = rows.dropna(subset=["lev"]).sort_values("lev")
            src_levels = rows["lev"].to_numpy(dtype=np.float32)
            nearest = np.abs(src_levels[:, None] - self.levels[None, :]).argmin(axis=0)
            self.stats[str(name)] = {
                "has_lev": True,
                "mean": rows["mean"].to_numpy(dtype=np.float32)[nearest],
                "std": rows["std"].to_numpy(dtype=np.float32)[nearest],
                "min": rows["min"].to_numpy(dtype=np.float32)[nearest],
                "max": rows["max"].to_numpy(dtype=np.float32)[nearest],
                "lev": self.levels.copy(),
            }

    def _values(self, name: str, key: str, channels: int) -> np.ndarray:
        stat = self.stats[name]
        value = stat[key]
        if bool(stat["has_lev"]):
            out = np.asarray(value, dtype=np.float32)
        else:
            out = np.full(channels, float(value), dtype=np.float32)
        if out.size != channels:
            raise ValueError(f"{name}: statistics have {out.size} channels, data have {channels}")
        return out

    @staticmethod
    def _broadcast(values: np.ndarray, ndim: int) -> np.ndarray:
        return values.reshape((1,) * (ndim - 3) + (values.size, 1, 1))

    def normalize(self, name: str, array: np.ndarray, *, atmosphere: bool = False) -> np.ndarray:
        x = np.asarray(array, dtype=np.float32)
        channels = int(x.shape[-3])
        vmin = self._broadcast(self._values(name, "min", channels), x.ndim)
        vmax = self._broadcast(self._values(name, "max", channels), x.ndim)
        eps = 1e-6

        minmax = MINMAX_ATM_VARS if atmosphere else MINMAX_STATE_VARS
        symmetric = SYMMETRIC_ATM_VARS if atmosphere else SYMMETRIC_STATE_VARS
        if name in minmax:
            return ((x - vmin) / (vmax - vmin + eps)).astype(np.float32)
        if name in symmetric:
            scale = np.maximum(np.abs(vmin), np.abs(vmax))
            return (x / (scale + eps)).astype(np.float32)

        mean = self._broadcast(self._values(name, "mean", channels), x.ndim)
        std = self._broadcast(self._values(name, "std", channels), x.ndim)
        return ((x - mean) / (std + eps)).astype(np.float32)

    def denormalize(self, name: str, array: np.ndarray) -> np.ndarray:
        x = np.asarray(array, dtype=np.float32)
        channels = int(x.shape[-3])
        vmin = self._broadcast(self._values(name, "min", channels), x.ndim)
        vmax = self._broadcast(self._values(name, "max", channels), x.ndim)
        eps = 1e-6

        if name in MINMAX_STATE_VARS:
            return (x * (vmax - vmin + eps) + vmin).astype(np.float32)
        if name in SYMMETRIC_STATE_VARS:
            scale = np.maximum(np.abs(vmin), np.abs(vmax))
            return (x * (scale + eps)).astype(np.float32)

        mean = self._broadcast(self._values(name, "mean", channels), x.ndim)
        std = self._broadcast(self._values(name, "std", channels), x.ndim)
        return (x * (std + eps) + mean).astype(np.float32)


class OceanDatasetBase(Dataset):
    """Shared Zarr reader and channel metadata."""

    def __init__(
        self,
        data_dir: str,
        rollout_days: int,
        warmup_days: int,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cache_days: int = 10,
        norm_csv: str | Path = DEFAULT_NORM_CSV,
    ):
        if rollout_days < 1 or warmup_days < 0:
            raise ValueError("rollout_days must be >= 1 and warmup_days must be >= 0")

        self.data_dir = str(data_dir)
        self.rollout_days = int(rollout_days)
        self.warmup_days = int(warmup_days)
        self.cache_days = max(int(cache_days), 0)
        self._zarr: Optional[xr.Dataset] = None
        self._cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()

        ds = self._open_zarr()
        missing = [v for v in (*STATE_VARS, *ATM_VARS, "cos_sza_utc00") if v not in ds]
        if missing:
            raise KeyError(f"Zarr store is missing variables: {missing}")
        if "time" not in ds.coords or "lev" not in ds.coords:
            raise KeyError("Zarr store must contain time and lev coordinates")

        self.file_dates = [_time_token(v) for v in np.asarray(ds.time.values)]
        self.levels = np.asarray(ds.lev.values, dtype=np.float32)
        self.lat_values = np.asarray(ds.lat.values, dtype=np.float32)
        self.lon_values = np.asarray(ds.lon.values, dtype=np.float32)
        self.normalizer = Normalizer(norm_csv, self.levels)
        self.norm_stats = self.normalizer.stats

        self.y_vars = list(STATE_VARS)
        self.atm_vars = list(ATM_VARS)
        self.aux_vars = ["land_mask", "cos_sza_utc00"]
        self.y_var_slices: Dict[str, slice] = {}
        offset = 0
        for name in STATE_VARS:
            channels = len(self.levels) if name in {"thetao", "so", "uo", "vo"} else 1
            self.y_var_slices[name] = slice(offset, offset + channels)
            offset += channels
        self.n_state_channels = offset

        start_dt = _to_datetime(start_date) if start_date else None
        end_dt = _to_datetime(end_date) if end_date else None
        latest = len(self.file_dates) - self.rollout_days - 1
        valid = []
        for index in range(self.warmup_days, latest + 1):
            first = _to_datetime(self.file_dates[index + 1])
            last = _to_datetime(self.file_dates[index + self.rollout_days])
            if start_dt and first < start_dt:
                continue
            if end_dt and last > end_dt:
                continue
            valid.append(index)
        if not valid:
            raise ValueError("No samples remain after date and rollout-horizon filtering")
        self.valid_start_indices = valid

        sample_raw = self._read_raw_state(valid[0])
        surface_valid = np.isfinite(sample_raw["uo"][0]) & np.isfinite(sample_raw["vo"][0])
        self.surface_land_mask = (~surface_valid).astype(np.float32)[None]
        masks = []
        for name in STATE_VARS:
            finite = np.isfinite(sample_raw[name])
            masks.append((finite & surface_valid[None]).astype(np.float32))
        self.y_valid_mask = np.concatenate(masks, axis=0)
        self.state_idx_in_x = np.arange(self.n_state_channels, dtype=np.int64)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_zarr"] = None
        state["_cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return len(self.valid_start_indices)

    def _open_zarr(self) -> xr.Dataset:
        if self._zarr is None:
            self._zarr = xr.open_zarr(self.data_dir, consolidated=True, chunks=None)
        return self._zarr

    def _read_raw_state(self, day: int) -> Dict[str, np.ndarray]:
        ds = self._open_zarr().isel(time=int(day), drop=True)
        out = {name: as_chw(ds[name]) for name in STATE_VARS}
        out["siconc"] = np.clip(out["siconc"], 0.0, 1.0)
        out["sithick"] = np.clip(out["sithick"], 0.0, None)
        return out

    @staticmethod
    def _land_mask(raw: Dict[str, np.ndarray]) -> np.ndarray:
        ocean = np.isfinite(raw["uo"][0]) & np.isfinite(raw["vo"][0])
        return (~ocean).astype(np.float32)[None]

    def _load_day(self, day: int) -> Dict[str, np.ndarray]:
        day = int(day)
        if day in self._cache:
            self._cache.move_to_end(day)
            return self._cache[day]

        ds = self._open_zarr().isel(time=day, drop=True)
        raw = self._read_raw_state(day)
        state = np.concatenate(
            [self.normalizer.normalize(name, raw[name]) for name in STATE_VARS], axis=0
        )
        aux = np.concatenate(
            [self._land_mask(raw), as_chw(ds["cos_sza_utc00"])], axis=0
        )
        atmosphere = np.concatenate(
            [self.normalizer.normalize(name, as_chw(ds[name]), atmosphere=True) for name in ATM_VARS],
            axis=0,
        )
        bundle = {
            "state": state.astype(np.float16),
            "aux": aux.astype(np.float16),
            "atmosphere": atmosphere.astype(np.float16),
        }
        if self.cache_days:
            self._cache[day] = bundle
            self._cache.move_to_end(day)
            while len(self._cache) > self.cache_days:
                self._cache.popitem(last=False)
        return bundle

    def input_for_day(self, day: int) -> np.ndarray:
        bundle = self._load_day(day)
        return np.concatenate(
            [bundle["state"], bundle["aux"], bundle["atmosphere"]], axis=0
        ).astype(np.float32)

    def aux_for_day(self, day: int) -> np.ndarray:
        day = int(day)
        if day in self._cache:
            return np.asarray(self._cache[day]["aux"], dtype=np.float32)
        ds = self._open_zarr().isel(time=day, drop=True)
        return np.concatenate(
            [self.surface_land_mask, as_chw(ds["cos_sza_utc00"])], axis=0
        ).astype(np.float32)

    @staticmethod
    def _stack(items, channels: int, height: int, width: int) -> np.ndarray:
        if items:
            return np.stack(items).astype(np.float32)
        return np.zeros((0, channels, height, width), dtype=np.float32)

    @staticmethod
    def _tensor(array: np.ndarray) -> torch.Tensor:
        clean = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(clean.astype(np.float32, copy=False))

    def _initial_fields(self, start: int) -> Dict[str, torch.Tensor]:
        x0 = self.input_for_day(start)
        height, width = x0.shape[-2:]
        warmup = [self.input_for_day(i) for i in range(start - self.warmup_days, start)]
        aux = [self.aux_for_day(start + k) for k in range(1, self.rollout_days)]
        return {
            "warmup_inputs": self._tensor(self._stack(warmup, x0.shape[0], height, width)),
            "input": self._tensor(x0),
            "aux_future": self._tensor(self._stack(aux, 2, height, width)),
            "start_index": torch.tensor(start, dtype=torch.long),
        }

    def valid_mask_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.from_numpy(self.y_valid_mask[None]).to(device=device)

    def latitude_weights_tensor(self, floor: float = 0.2) -> torch.Tensor:
        phi = np.deg2rad(np.abs(self.lat_values))
        polar = floor + (1.0 - floor) * np.abs(np.sin(phi))
        ocean = floor + (1.0 - floor) * np.abs(np.cos(phi))
        polar = polar / polar.mean()
        ocean = ocean / ocean.mean()
        blocks = []
        for name in STATE_VARS:
            channels = self.y_var_slices[name].stop - self.y_var_slices[name].start
            profile = polar if name in {"siconc", "sithick"} else ocean
            blocks.append(np.broadcast_to(profile[None, :, None], (channels, len(profile), 1)))
        return torch.from_numpy(np.concatenate(blocks).astype(np.float32)[None])

    def denormalize(self, array: np.ndarray) -> np.ndarray:
        """Convert ``(..., channel, lat, lon)`` model values to physical units."""
        out = np.empty_like(array, dtype=np.float32)
        for name, channel_slice in self.y_var_slices.items():
            out[..., channel_slice, :, :] = self.normalizer.denormalize(
                name, array[..., channel_slice, :, :]
            )
        return out


class RolloutDataset(OceanDatasetBase):
    """Observed targets and ERA5 forcing for training or hindcast inference."""

    def __getitem__(self, item: int):
        start = self.valid_start_indices[item]
        sample = self._initial_fields(start)
        height, width = sample["input"].shape[-2:]
        forcing = [
            np.asarray(self._load_day(start + k)["atmosphere"], dtype=np.float32)
            for k in range(1, self.rollout_days)
        ]
        targets = [
            np.asarray(self._load_day(start + k)["state"], dtype=np.float32)
            for k in range(1, self.rollout_days + 1)
        ]
        sample["forcing_future"] = self._tensor(
            self._stack(forcing, len(ATM_VARS), height, width)
        )
        sample["y_future"] = self._tensor(np.stack(targets).astype(np.float32))
        return sample


class InitialConditionDataset(OceanDatasetBase):
    """Initial state, warm-up, and future auxiliary fields for SEAS5 forecasts."""

    def __getitem__(self, item: int):
        return self._initial_fields(self.valid_start_indices[item])
