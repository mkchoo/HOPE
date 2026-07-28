"""HOPE hierarchical recurrent ocean-cryosphere emulator."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .LightUnetPP import LayerNorm2d, LightUNetPlusPlus
from .swin2sr_lonshift import Swin2SR_DFE
from .utils import (
    CnnEmbeddingLayer,
    build_latitude_grid_features,
    crop_to_original,
    lcm_of_list,
    pad_to_multiple,
    pad_to_multiple_calculator,
)


RolloutState = Sequence[torch.Tensor]


class HOPE(nn.Module):
    """One-day transition model with persistent low/high latent states."""

    def __init__(self, config):
        super().__init__()
        self.T = int(config.hope_t)
        self.C = int(config.hope_c)
        self.out_chans = int(config.out_chans)
        self.z_dims = int(config.z_dims)

        self.embed_layer = CnnEmbeddingLayer(config.in_chans, config.embed_dims)
        self.grid_proj = nn.Conv2d(2, config.embed_dims, kernel_size=1, bias=False)
        nn.init.zeros_(self.grid_proj.weight)
        self._grid_cache: Dict[Tuple[int, int, str, Optional[int], str], torch.Tensor] = {}

        low = config.LightUnetPP
        self.L_net_model = LightUNetPlusPlus(
            in_channels=config.embed_dims,
            z_channels=config.z_dims,
            out_channels=config.z_dims,
            init_features=low["init_features"],
            num_groups=low["num_groups"],
            convnext_num_blocks=low["convnext_num_blocks"],
            convnext_expansion=low["convnext_expansion"],
            convnext_kernel_size=low["convnext_kernel_size"],
            convnext_layer_scale_init_value=low["convnext_layer_scale_init_value"],
            convnext_drop_path=low["convnext_drop_path"],
            lon_periodic=True,
            spherical_lat=True,
            pad_mode="reflect",
        )

        high = config.Swin2SR
        self.padding_multiply_factor = lcm_of_list(
            [self.L_net_model.padding_multiply_factor, high["window_size"]]
        )
        (_, _), (padded_height, padded_width) = pad_to_multiple_calculator(
            config.height, config.width, self.padding_multiply_factor
        )
        self.H_net_model = Swin2SR_DFE(
            img_size=(padded_height, padded_width),
            patch_size=1,
            embed_dim=config.z_dims,
            depths=high["depths"],
            num_heads=high["num_heads"],
            window_size=high["window_size"],
            mlp_ratio=high["mlp_ratio"],
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            img_range=1.0,
            resi_connection="1conv",
            shift_axis="x",
            lon_periodic=True,
            spherical_lat=True,
            pad_mode="reflect",
        )

        self.head_layer = CnnEmbeddingLayer(config.z_dims, config.out_chans)
        with torch.no_grad():
            self.head_layer.net[0].weight.mul_(1e-2)
            self.head_layer.net[0].bias.zero_()

        # These three frozen tensors are inert, but retain exact compatibility
        # with the released checkpoints.
        self.head_delta_scale = nn.Parameter(torch.ones(config.out_chans), requires_grad=False)
        update_logit = math.log(0.25 / 0.75)
        self.zL_logit = nn.Parameter(torch.tensor(update_logit), requires_grad=False)
        self.zH_logit = nn.Parameter(torch.tensor(update_logit), requires_grad=False)

        high_init = torch.empty(1, config.z_dims, 1, 1)
        low_init = torch.empty(1, config.z_dims, 1, 1)
        nn.init.trunc_normal_(high_init, std=1.0)
        nn.init.trunc_normal_(low_init, std=1.0)
        self.register_buffer("H_init", high_init, persistent=True)
        self.register_buffer("L_init", low_init, persistent=True)
        self.zH_norm = LayerNorm2d(config.z_dims)
        self.zL_norm = LayerNorm2d(config.z_dims)
        self.z_norm = nn.Identity()

    def _grid(self, height, width, device, dtype):
        key = (int(height), int(width), device.type, device.index, str(dtype))
        if key not in self._grid_cache:
            self._grid_cache[key] = build_latitude_grid_features(
                height, width, device=device, dtype=dtype
            )
        return self._grid_cache[key]

    def L_net(self, embedding: torch.Tensor, zL: torch.Tensor, zH: torch.Tensor) -> torch.Tensor:
        return self.zL_norm(zL + self.L_net_model(embedding, zL + zH))

    def H_net(self, zH: torch.Tensor, zL: torch.Tensor) -> torch.Tensor:
        return self.zH_norm(self.H_net_model(zH + zL))

    def _unroll(self, embedding, zH, zL, steps: int, start_step: int = 0):
        for offset in range(steps):
            step = start_step + offset
            zL = self.L_net(embedding, zL, zH)
            if (step + 1) % self.T == 0:
                zH = self.H_net(zH, zL)
        return zH, zL

    def _prepare(
        self,
        x: torch.Tensor,
        state: Optional[RolloutState],
    ):
        batch, _, height, width = x.shape
        original_size = (height, width)
        x_pad, _ = pad_to_multiple(
            x,
            self.padding_multiply_factor,
            lon_periodic=True,
            spherical_lat=True,
            mode="reflect",
        )
        embedding = self.embed_layer(x_pad)
        embedding = embedding + self.grid_proj(
            self._grid(x_pad.shape[-2], x_pad.shape[-1], x.device, x.dtype)
        )

        if state is None:
            zH = self.H_init.to(x).expand(batch, -1, height, width).contiguous()
            zL = self.L_init.to(x).expand(batch, -1, height, width).contiguous()
        else:
            if len(state) != 2:
                raise ValueError("Rollout state must be (zH, zL)")
            zH, zL = state
            if zH.shape[-2:] != original_size or zL.shape[-2:] != original_size:
                raise ValueError("Latent state grid does not match the input grid")
            zH = zH.to(x)
            zL = zL.to(x)

        zH, _ = pad_to_multiple(
            zH,
            self.padding_multiply_factor,
            lon_periodic=True,
            spherical_lat=True,
            mode="reflect",
        )
        zL, _ = pad_to_multiple(
            zL,
            self.padding_multiply_factor,
            lon_periodic=True,
            spherical_lat=True,
            mode="reflect",
        )
        return x_pad, embedding, zH, zL, original_size

    def _finish(self, x_pad, zH, zL, original_size):
        delta = self.head_layer(zH)
        prediction = x_pad[:, : self.out_chans] + delta
        return (
            crop_to_original(zH, original_size),
            crop_to_original(zL, original_size),
        ), crop_to_original(prediction, original_size)

    def forward(self, x: torch.Tensor, state: Optional[RolloutState] = None):
        """Train one transition; gradients pass through the final inner update."""
        x_pad, embedding, zH, zL, original_size = self._prepare(x, state)
        total_steps = self.T * self.C
        with torch.no_grad():
            zH, zL = self._unroll(embedding.detach(), zH, zL, total_steps - 1)
        zH, zL = self._unroll(
            embedding, zH.detach(), zL.detach(), steps=1, start_step=total_steps - 1
        )
        return self._finish(x_pad, zH, zL, original_size)

    @torch.inference_mode()
    def spinup_state(self, x: torch.Tensor, state: Optional[RolloutState] = None):
        """Advance zH/zL using an observed day without producing a forecast."""
        _, embedding, zH, zL, original_size = self._prepare(x, state)
        zH, zL = self._unroll(embedding, zH, zL, self.T * self.C)
        return crop_to_original(zH, original_size), crop_to_original(zL, original_size)
