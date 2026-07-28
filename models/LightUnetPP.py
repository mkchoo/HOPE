import torch
import torch.nn as nn

from .utils import GeoConv2d


# -----------------------------------------------------------------------------
# ConvNeXt building blocks (2D, channels-first)
# -----------------------------------------------------------------------------


def _drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth (per-sample)."""
    if drop_prob == 0.0 or (not training):
        return x
    keep_prob = 1.0 - float(drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = random_tensor.floor()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for (B,C,H,W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # (B,H,W,C)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # (B,C,H,W)


class ConvNeXtBlock2d(nn.Module):
    """ConvNeXt block for channels-first tensors with geo-aware depthwise conv."""

    def __init__(
        self,
        dim: int,
        *,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale_init_value: float = 1e-6,
        drop_path: float = 0.0,
        lon_periodic: bool = True,
        spherical_lat: bool = True,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        self.dwconv = GeoConv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            lon_periodic=lon_periodic,
            spherical_lat=spherical_lat,
            pad_mode=pad_mode,
        )
        self.norm = LayerNorm2d(dim)
        hidden_dim = int(dim * expansion)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = self.drop_path(x)
        return shortcut + x


class ConvNeXtConvBlock(nn.Module):
    """UNet-style block built from ConvNeXt blocks.

    First does a 1x1 projection (if needed) then applies `num_blocks` ConvNeXt blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_blocks: int = 2,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale_init_value: float = 1e-6,
        drop_path: float = 0.0,
        lon_periodic: bool = True,
        spherical_lat: bool = True,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        blocks = []
        for _ in range(int(num_blocks)):
            blocks.append(
                ConvNeXtBlock2d(
                    out_channels,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    layer_scale_init_value=layer_scale_init_value,
                    drop_path=drop_path,
                    lon_periodic=lon_periodic,
                    spherical_lat=spherical_lat,
                    pad_mode=pad_mode,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.blocks(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, lon_periodic=True, spherical_lat=True, pad_mode="reflect"):
        super().__init__()
        self.conv = nn.Sequential(
            GeoConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                lon_periodic=lon_periodic,
                spherical_lat=spherical_lat,
                pad_mode=pad_mode,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            GeoConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                lon_periodic=lon_periodic,
                spherical_lat=spherical_lat,
                pad_mode=pad_mode,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlockGroupNorm(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=16, *, lon_periodic=True, spherical_lat=True, pad_mode="reflect"):
        super().__init__()
        self.conv = nn.Sequential(
            GeoConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                lon_periodic=lon_periodic,
                spherical_lat=spherical_lat,
                pad_mode=pad_mode,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(inplace=True),
            GeoConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                lon_periodic=lon_periodic,
                spherical_lat=spherical_lat,
                pad_mode=pad_mode,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LightUNetPlusPlus(nn.Module):
    def __init__(
        self,
        in_channels=3,
        z_channels=64,
        out_channels=64,
        init_features=32,
        num_groups=16,
        convnext_num_blocks: int = 2,
        convnext_expansion: int = 4,
        convnext_kernel_size: int = 7,
        convnext_layer_scale_init_value: float = 1e-6,
        convnext_drop_path: float = 0.0,
        lon_periodic: bool = True,
        spherical_lat: bool = True,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        features = int(init_features)
        self.padding_multiply_factor = 4

        input_dim = int(in_channels + z_channels)
        nb = int(convnext_num_blocks)
        exp = int(convnext_expansion)
        ks = int(convnext_kernel_size)

        block_kwargs = dict(
            num_blocks=nb,
            kernel_size=ks,
            expansion=exp,
            layer_scale_init_value=float(convnext_layer_scale_init_value),
            drop_path=float(convnext_drop_path),
            lon_periodic=lon_periodic,
            spherical_lat=spherical_lat,
            pad_mode=pad_mode,
        )

        # Encoder
        self.conv0_0 = ConvNeXtConvBlock(input_dim, features, **block_kwargs)
        self.conv1_0 = ConvNeXtConvBlock(features, features * 2, **block_kwargs)
        self.conv2_0 = ConvNeXtConvBlock(features * 2, features * 4, **block_kwargs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Nested skip pathways
        self.conv0_1 = ConvNeXtConvBlock(features + features * 2, features, **block_kwargs)
        self.conv1_1 = ConvNeXtConvBlock(features * 2 + features * 4, features * 2, **block_kwargs)
        self.conv0_2 = ConvNeXtConvBlock(features + features + features * 2, features, **block_kwargs)

        # Final output
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x, z):
        concat_x_z = torch.cat((x, z), dim=1)
        x0_0 = self.conv0_0(concat_x_z)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        return self.final_conv(x0_2)
