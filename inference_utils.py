"""Shared model loading, rollout, and NetCDF output for HOPE inference."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import netCDF4
import numpy as np
import torch

from dataloader import OceanDatasetBase
from exp_config.config import Config
from models.hope import HOPE


def parse_time(token: str) -> dt.datetime:
    return dt.datetime.strptime(str(token), "%Y%m%d%H%M")


def load_ema_model(checkpoint: str, dataset: OceanDatasetBase, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "ema_model" not in payload:
        raise KeyError(f"Checkpoint has no ema_model: {checkpoint}")
    config = Config(
        name="inference",
        height=len(dataset.lat_values),
        width=len(dataset.lon_values),
        in_chans=dataset.n_state_channels + 2 + 8,
        out_chans=dataset.n_state_channels,
    ).sync_dims()
    model = HOPE(config)
    model.load_state_dict(payload["ema_model"], strict=True)
    return model.to(device).eval()


def observed_warmup(model, inputs: torch.Tensor, device: torch.device):
    state = None
    for day in range(inputs.shape[1]):
        state = model.spinup_state(inputs[:, day].to(device, non_blocking=True), state)
    return state


@torch.inference_mode()
def rollout(
    model,
    x0: torch.Tensor,
    warmup_inputs: torch.Tensor,
    auxiliary: torch.Tensor,
    forcing: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Autoregress with prescribed future atmosphere and auxiliary fields."""
    device = next(model.parameters()).device
    x = x0.to(device, non_blocking=True)
    auxiliary = auxiliary.to(device, non_blocking=True)
    forcing = forcing.to(device, non_blocking=True)
    mask = valid_mask.to(device)
    state = observed_warmup(model, warmup_inputs, device)

    predictions = []
    rollout_days = int(forcing.shape[1]) + 1
    for lead in range(rollout_days):
        state, prediction = model(x, state)
        prediction = prediction * mask.to(prediction.dtype)
        predictions.append(prediction)
        if lead + 1 < rollout_days:
            x = torch.cat([prediction, auxiliary[:, lead], forcing[:, lead]], dim=1)
    return torch.stack(predictions, dim=1)


def to_physical(dataset: OceanDatasetBase, normalized: np.ndarray) -> np.ndarray:
    raw = dataset.denormalize(np.asarray(normalized, dtype=np.float32))
    valid = dataset.y_valid_mask.astype(bool)
    return np.where(valid[None], raw, np.nan).astype(np.float32)


def _write_group(root, name: str, array: np.ndarray, dataset: OceanDatasetBase) -> None:
    group = root.createGroup(name)
    for variable, channel_slice in dataset.y_var_slices.items():
        block = np.asarray(array[:, channel_slice], dtype=np.float32)
        channels = int(block.shape[1])
        if channels == 1:
            dims = ("lead", "lat", "lon")
            values = block[:, 0]
        else:
            level_dim = f"{variable}_lev"
            if level_dim not in root.dimensions:
                root.createDimension(level_dim, channels)
                level = root.createVariable(level_dim, "f4", (level_dim,))
                level[:] = dataset.levels
                level.units = "m"
            dims = ("lead", level_dim, "lat", "lon")
            values = block

        field = group.createVariable(
            variable,
            "f4",
            dims,
            fill_value=np.float32(np.nan),
            zlib=True,
            complevel=6,
            shuffle=True,
        )
        field[:] = values


def save_netcdf(
    path: str | Path,
    dataset: OceanDatasetBase,
    prediction: np.ndarray,
    *,
    truth: Optional[np.ndarray] = None,
    init_time: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as root:
        leads, _, height, width = prediction.shape
        root.createDimension("lead", leads)
        root.createDimension("lat", height)
        root.createDimension("lon", width)

        lead = root.createVariable("lead", "i4", ("lead",))
        lead[:] = np.arange(1, leads + 1)
        lead.units = "days"
        lat = root.createVariable("lat", "f4", ("lat",))
        lat[:] = dataset.lat_values
        lat.units = "degrees_north"
        lon = root.createVariable("lon", "f4", ("lon",))
        lon[:] = dataset.lon_values
        lon.units = "degrees_east"

        root.description = "HOPE ocean-cryosphere rollout in physical units"
        root.init_time = str(init_time)
        _write_group(root, "pred", prediction, dataset)
        if truth is not None:
            _write_group(root, "true", truth, dataset)
