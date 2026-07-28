"""Fixed architecture used by the released HOPE experiment."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class Config:
    name: str = "hope"
    height: int = 180
    width: int = 360
    in_chans: int = 105
    out_chans: int = 95

    # Hierarchical recurrence: two low-level updates per cycle, two cycles.
    hope_t: int = 2
    hope_c: int = 2
    embed_dims: int = 128
    z_dims: int = 128

    LightUnetPP: Dict[str, Any] = field(
        default_factory=lambda: {
            "init_features": 64,
            "num_groups": 16,
            "convnext_num_blocks": 2,
            "convnext_expansion": 4,
            "convnext_kernel_size": 7,
            "convnext_layer_scale_init_value": 1e-6,
            "convnext_drop_path": 0.0,
            "lon_periodic": True,
            "spherical_lat": True,
            "pad_mode": "reflect",
        }
    )
    Swin2SR: Dict[str, Any] = field(
        default_factory=lambda: {
            "img_size": (180, 360),
            "patch_size": 1,
            "embed_dim": 128,
            "depths": [2, 4, 2],
            "num_heads": [8, 8, 8],
            "window_size": 6,
            "mlp_ratio": 2.0,
            "qkv_bias": True,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "ape": False,
            "patch_norm": True,
            "resi_connection": "1conv",
            "shift_axis": "x",
            "lon_periodic": True,
            "spherical_lat": True,
            "use_checkpoint": False,
            "img_range": 1.0,
            "pad_mode": "reflect",
        }
    )

    def sync_dims(self) -> "Config":
        self.Swin2SR = dict(self.Swin2SR)
        self.Swin2SR["img_size"] = (self.height, self.width)
        self.Swin2SR["embed_dim"] = self.z_dims
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
