# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lightweight fixed-SRM spatial guidance for final F2 features.

These blocks are intentionally simple: SRM is used only to produce a 1-channel
spatial guidance mask, while the detector content remains the normal post-FPN F2
feature. This keeps the RGB+S baseline path untouched when the zero-initialized
residual scale is zero.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _group_norm_groups(channels: int) -> int:
    """Largest divisor <= 8 for lightweight branch normalization."""
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


def _grayscale(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    return (image * weights).sum(1, keepdim=True)


def _saturation(image: torch.Tensor) -> torch.Tensor:
    """Parameter-free HSV saturation from the raw RGB tap."""
    maximum = image.max(dim=1, keepdim=True).values
    minimum = image.min(dim=1, keepdim=True).values
    return torch.where(
        maximum > 0,
        (maximum - minimum) / maximum.clamp_min(torch.finfo(image.dtype).eps),
        torch.zeros_like(maximum),
    )


# Canonical compact SRM-3 set commonly reused by SRM-based forensic networks.
# Denominators are baked into the tensors to preserve the standard scale.
_CANONICAL_SRM3 = torch.tensor(
    [
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [0, -1, 4, -1, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    ],
    dtype=torch.float32,
)
_CANONICAL_SRM3[0] /= 4.0
_CANONICAL_SRM3[1] /= 12.0
_CANONICAL_SRM3[2] /= 2.0

# Existing LEVIR train-split directional winners used elsewhere in this repo.
_K4_K12_K16 = torch.tensor(
    [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, -2, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, -3, 0, 0],
            [0, 0, 0, 3, 0],
            [0, 0, 0, 0, -1],
        ],
    ],
    dtype=torch.float32,
)


class SRMF2Guidance(nn.Module):
    """Use fixed SRM residuals only as a spatial guidance mask on final F2.

    Input order: ``[F2_base, raw_rgb]``.

    ``F2_out = F2_base + gamma * mask * F2_base``

    Args:
        in_channels: Automatically injected channel list ``[F2_channels, 3]``.
        kernel_set: ``"srm3"`` or ``"k4k12k16"``.
        hidden_channels: Width of the tiny SRM encoder.
        dropout: Dropout2d probability inside the SRM branch.
        gamma_max: If > 0, use bounded scale ``gamma_max * tanh(gamma_raw)``.
        debug: Return ``(out, mask)`` instead of just ``out``.
        use_saturation_guidance: Add a separate saturation-mask branch and
            zero-initialized classification residual scale.
    """

    def __init__(
        self,
        in_channels: list[int],
        kernel_set: str = "srm3",
        hidden_channels: int = 32,
        dropout: float = 0.0,
        gamma_max: float = 0.0,
        debug: bool = False,
        use_saturation_guidance: bool = False,
    ) -> None:
        super().__init__()
        if len(in_channels) != 2:
            raise ValueError("SRMF2Guidance expects [F2_base, raw_rgb]")
        f2_channels, image_channels = in_channels
        if image_channels != 3:
            raise ValueError(f"raw RGB tap must have 3 channels, got {image_channels}")
        if kernel_set == "srm3":
            kernels = _CANONICAL_SRM3.clone()
        elif kernel_set == "k4k12k16":
            kernels = _K4_K12_K16.clone()
        else:
            raise ValueError(f"unknown kernel_set={kernel_set!r}")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be > 0")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        self.f2_channels = int(f2_channels)
        self.kernel_set = kernel_set
        self.gamma_max = float(gamma_max)
        self.debug = bool(debug)
        self.use_saturation_guidance = bool(use_saturation_guidance)
        self.register_buffer("srm_kernels", kernels.unsqueeze(1), persistent=True)  # [3,1,5,5]

        hidden_channels = int(hidden_channels)
        branch = [
            nn.Conv2d(3, hidden_channels, 1, bias=False),
            nn.GroupNorm(_group_norm_groups(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
            nn.GroupNorm(_group_norm_groups(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            branch.append(nn.Dropout2d(float(dropout)))
        branch.append(nn.Conv2d(hidden_channels, 1, 1))
        self.mask_net = nn.Sequential(*branch)
        self.gamma_raw = nn.Parameter(torch.zeros(()))
        if self.use_saturation_guidance:
            self.saturation_mask_net = nn.Sequential(
                nn.Conv2d(1, hidden_channels, 1, bias=False),
                nn.GroupNorm(_group_norm_groups(hidden_channels), hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
                nn.GroupNorm(_group_norm_groups(hidden_channels), hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, 1),
            )
            self.gamma_saturation_raw = nn.Parameter(torch.zeros(()))

        self.last_mask = None
        self.last_gamma = None
        self.last_residuals = None
        self.last_saturation_mask = None
        self.last_saturation_gamma = None
        self.current_mask = None  # legacy diagnostic slot; must never retain a graph for EMA deepcopy

    def effective_gamma(self) -> torch.Tensor:
        if self.gamma_max > 0:
            return self.gamma_max * torch.tanh(self.gamma_raw)
        return self.gamma_raw

    def make_mask(self, f2: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Return the graph-connected SRM mask without caching it on the module."""
        gray = _grayscale(image)
        kernels = self.srm_kernels.to(dtype=gray.dtype)
        residuals = F.conv2d(gray, kernels, padding=2).abs()
        if residuals.shape[-2:] != f2.shape[-2:]:
            # Area downsampling preserves local response energy better than point sampling.
            residuals = F.interpolate(residuals, size=f2.shape[-2:], mode="area")
        mask = torch.sigmoid(self.mask_net(residuals))
        self.last_residuals = residuals.detach()
        return mask

    def forward(self, inputs):
        f2, image = inputs
        mask = self.make_mask(f2, image)
        gamma = self.effective_gamma()
        out = f2 + gamma * mask * f2
        if self.use_saturation_guidance:
            saturation = _saturation(image)
            if saturation.shape[-2:] != f2.shape[-2:]:
                saturation = F.interpolate(saturation, size=f2.shape[-2:], mode="area")
            saturation_mask = torch.sigmoid(self.saturation_mask_net(saturation))
            out = out + self.gamma_saturation_raw * saturation_mask * f2
            self.last_saturation_mask = saturation_mask.detach()
            self.last_saturation_gamma = self.gamma_saturation_raw.detach()

        self.last_mask = mask.detach()
        self.current_mask = None
        self.last_gamma = gamma.detach()
        return (out, mask) if self.debug else out


__all__ = ("SRMF2Guidance",)
