"""Geometry and loss helpers used by HOPE."""

from __future__ import annotations

import math
from functools import reduce
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnEmbeddingLayer(nn.Module):
    def __init__(self, in_channels: int, output_channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, output_channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def spherical_pad_bottom_latitude(x: torch.Tensor, rows: int) -> torch.Tensor:
    if rows <= 0:
        return x
    opposite = torch.roll(x, shifts=x.shape[-1] // 2, dims=-1)
    return torch.cat([x, torch.flip(opposite[:, :, -rows:, :], dims=[-2])], dim=-2)


def spherical_pad_latitude(
    x: torch.Tensor, top: int = 0, bottom: int = 0
) -> torch.Tensor:
    if top <= 0 and bottom <= 0:
        return x
    opposite = torch.roll(x, shifts=x.shape[-1] // 2, dims=-1)
    parts = []
    if top:
        parts.append(torch.flip(opposite[:, :, :top, :], dims=[-2]))
    parts.append(x)
    if bottom:
        parts.append(torch.flip(opposite[:, :, -bottom:, :], dims=[-2]))
    return torch.cat(parts, dim=-2)


def geo_pad2d(
    x: torch.Tensor,
    pad: Sequence[int],
    *,
    lon_periodic: bool = True,
    spherical_lat: bool = False,
    mode: str = "reflect",
) -> torch.Tensor:
    left, right, top, bottom = [int(value) for value in pad]
    if top or bottom:
        if spherical_lat:
            x = spherical_pad_latitude(x, top, bottom)
        else:
            x = F.pad(x, (0, 0, top, bottom), mode=mode)
    if left or right:
        x = F.pad(x, (left, right, 0, 0), mode="circular" if lon_periodic else mode)
    return x


class GeoConv2d(nn.Conv2d):
    """Conv2d with periodic longitude and pole-aware latitude padding."""

    def __init__(
        self,
        *args,
        lon_periodic: bool = True,
        spherical_lat: bool = False,
        pad_mode: str = "reflect",
        **kwargs,
    ):
        args = list(args)
        if "padding" in kwargs:
            padding = kwargs.pop("padding")
        elif len(args) >= 5:
            padding = args.pop(4)
        else:
            padding = 0
        super().__init__(*args, padding=0, **kwargs)
        if isinstance(padding, tuple):
            pad_h, pad_w = padding
        else:
            pad_h = pad_w = padding
        self.geo_padding = (int(pad_w), int(pad_w), int(pad_h), int(pad_h))
        self.lon_periodic = bool(lon_periodic)
        self.spherical_lat = bool(spherical_lat)
        self.pad_mode = str(pad_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(self.geo_padding):
            x = geo_pad2d(
                x,
                self.geo_padding,
                lon_periodic=self.lon_periodic,
                spherical_lat=self.spherical_lat,
                mode=self.pad_mode,
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )


def pad_to_multiple_calculator(
    height: int, width: int, multiple: int = 4
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    return (pad_h, pad_w), (height + pad_h, width + pad_w)


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = 4,
    *,
    lon_periodic: bool = False,
    spherical_lat: bool = False,
    mode: str = "reflect",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, height, width = tensor.shape
    (pad_h, pad_w), _ = pad_to_multiple_calculator(height, width, multiple)
    x = tensor
    if pad_w:
        x = F.pad(x, (0, pad_w, 0, 0), mode="circular" if lon_periodic else mode)
    if pad_h:
        if spherical_lat and lon_periodic:
            x = spherical_pad_bottom_latitude(x, pad_h)
        else:
            x = F.pad(x, (0, 0, 0, pad_h), mode=mode)
    return x, (height, width)


def crop_to_original(tensor: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    height, width = original_size
    return tensor[:, :, : int(height), : int(width)]


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def lcm_of_list(numbers) -> int:
    values = [int(value) for value in numbers if int(value)]
    return reduce(_lcm, values) if values else 1


def default_latitudes(height: int) -> np.ndarray:
    step = 180.0 / height
    return np.linspace(
        -90.0 + 0.5 * step,
        90.0 - 0.5 * step,
        height,
        dtype=np.float32,
    )


def build_latitude_grid_features(
    height: int,
    width: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    latitude = torch.from_numpy(default_latitudes(height)).to(
        device=device, dtype=dtype or torch.float32
    )
    latitude = torch.deg2rad(latitude)
    sin_lat = torch.sin(latitude).view(1, 1, height, 1).expand(1, 1, height, width)
    cos_lat = torch.cos(latitude).view(1, 1, height, 1).expand(1, 1, height, width)
    return torch.cat([sin_lat, cos_lat], dim=1)


def weighted_masked_mean(
    loss: torch.Tensor,
    mask: torch.Tensor,
    spatial_weight: torch.Tensor,
) -> torch.Tensor:
    weight = mask.to(loss) * spatial_weight.to(loss)
    weight = weight.expand_as(loss)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-12)
