"""Deterministic local-contrast and Scharr cues for clean input/guidance ablations."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def explicit_cue(rgb: torch.Tensor, cue: str, eps: float = 1e-6, window: int = 7) -> torch.Tensor:
    """Return one stable [B,1,H,W] cue from normal RGB in [0,1]."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"explicit cue expects RGB [B,3,H,W], got {tuple(rgb.shape)}")
    y = .299 * rgb[:, 0:1] + .587 * rgb[:, 1:2] + .114 * rgb[:, 2:3]
    if cue == "contrast":
        mu = F.avg_pool2d(y, window, 1, window // 2)
        return (torch.abs(y - mu) / (mu + eps)).clamp_(0, 5) / 5
    if cue == "local_chroma_contrast":
        # Chroma is the local colour spread (max RGB - min RGB).  Comparing it
        # with a neighbourhood average emphasizes coloured targets against
        # similarly bright but chromatically uniform background, while keeping
        # the cue deterministic and differentiable.
        chroma = rgb.amax(dim=1, keepdim=True) - rgb.amin(dim=1, keepdim=True)
        mu = F.avg_pool2d(chroma, window, 1, window // 2, count_include_pad=False)
        return (torch.abs(chroma - mu) / (mu + eps)).clamp_(0, 5) / 5
    if cue == "scharr":
        kx = y.new_tensor(((-3., 0., 3.), (-10., 0., 10.), (-3., 0., 3.))).view(1, 1, 3, 3)
        ky = y.new_tensor(((-3., -10., -3.), (0., 0., 0.), (3., 10., 3.))).view(1, 1, 3, 3)
        edge = torch.sqrt(F.conv2d(y, kx, padding=1).square() + F.conv2d(y, ky, padding=1).square() + eps)
        scale = edge.detach().mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return (edge / scale).clamp_(0, 5) / 5
    raise ValueError(f"Unsupported explicit cue: {cue}")


class RGBExplicitCue(nn.Module):
    """Append exactly one deterministic contrast/Scharr cue to RGB at input."""
    def __init__(self, cue: str) -> None:
        super().__init__()
        if cue not in {"contrast", "local_chroma_contrast", "scharr"}:
            raise ValueError("cue must be 'contrast', 'local_chroma_contrast', or 'scharr'")
        self.cue = cue
        self.out_channels = 4

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return torch.cat((rgb, explicit_cue(rgb, self.cue)), dim=1)


def _groups(channels: int) -> int:
    return next(group for group in (8, 4, 2, 1) if channels % group == 0)


class ExplicitCueGuidance(nn.Module):
    """Direct full-resolution cue-to-feature residual gated injection.

    Every level computes its own cue by resizing the same raw full-resolution
    map. No cue feature is passed between levels.
    """
    def __init__(self, channels: list[int] | tuple[int, int], cue: str, hidden: int = 32) -> None:
        super().__init__()
        feature_ch, rgb_ch = channels
        if rgb_ch != 3:
            raise ValueError("ExplicitCueGuidance requires an untouched RGB tap")
        self.cue = cue
        self.project = nn.Sequential(nn.Conv2d(1, feature_ch, 1, bias=False), nn.GroupNorm(_groups(feature_ch), feature_ch), nn.SiLU())
        self.gate = nn.Sequential(
            nn.Conv2d(2 * feature_ch, hidden, 1, bias=False), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False), nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(()))
        self.last_attention_stats: dict[str, float] = {}

    def forward(self, values: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feature, rgb = values
        cue = explicit_cue(rgb, self.cue)
        cue = F.interpolate(cue, size=feature.shape[-2:], mode="area")
        projected = self.project(cue)
        attention = self.gate(torch.cat((feature, projected), 1)).sigmoid()
        if not self.training:
            flat_attention = attention.detach().float().flatten(1)
            q = torch.quantile(flat_attention, flat_attention.new_tensor((.1, .25, .5, .75, .9)), dim=1)
            self.last_attention_stats = {
                "mean": float(flat_attention.mean()),
                "std": float(flat_attention.std()),
                "min": float(flat_attention.min()),
                "max": float(flat_attention.max()),
                **{f"p{p}": float(q[i].mean()) for i, p in enumerate((10, 25, 50, 75, 90))},
            }
        return feature + self.gamma * attention * projected


__all__ = ("RGBExplicitCue", "ExplicitCueGuidance", "explicit_cue")
