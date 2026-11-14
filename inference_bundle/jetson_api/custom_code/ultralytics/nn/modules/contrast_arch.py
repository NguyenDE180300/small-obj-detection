"""Contrast/SENetV2/EnSimAM/ASF architecture blocks for isolated ablations."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import ASFAttention, EnSimAM, ScalSeq
from .explicit_cues import explicit_cue


def _groups(channels: int) -> int:
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


class SENetV2(nn.Module):
    """Lightweight aggregated channel excitation with shape-preserving output."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.branches = nn.ModuleList(
            (
                nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.SiLU(), nn.Conv2d(hidden, channels, 1)),
                nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.SiLU(), nn.Conv2d(hidden, channels, 1)),
            )
        )
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, 1)
        excitation = sum(self.branches[i](pooled) for i in range(len(self.branches))) / len(self.branches)
        return x * torch.sigmoid(self.mix(excitation))


class LocalVarianceSimAM(nn.Module):
    """EnSimAM without its edge branch: global SimAM plus local variance.

    EnSimAM weights its global/local/edge branches as 0.5/0.25/0.25.
    Removing edge and renormalizing the retained weights gives 2/3 and 1/3,
    preserving the original relative contribution of the two kept branches.
    """

    def __init__(self, lambd: float = 1e-4, alpha: float = 1.0) -> None:
        super().__init__()
        self.lambd = float(lambd)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        energy = (x - mean).pow(2) / (4 * (var + self.lambd)) + 0.5
        global_attention = torch.sigmoid(1.0 / energy)

        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((x - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
        local_attention = torch.sigmoid(self.alpha * local_var)

        attention = (2.0 / 3.0) * global_attention + (1.0 / 3.0) * local_attention
        return x * attention


class SENetV2LocalVarianceSimAM(nn.Module):
    """Sequential SESA-style refinement: SENetV2 then LocalVarianceSimAM."""

    def __init__(self, channels: int, reduction: int = 16, lambd: float = 1e-4, alpha: float = 1.0) -> None:
        super().__init__()
        self.senet = SENetV2(channels, reduction)
        self.lv_simam = LocalVarianceSimAM(lambd=lambd, alpha=alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lv_simam(self.senet(x))


class ResidualSENetV2LocalVarianceSimAM(nn.Module):
    """Identity-initialized wrapper around the complete sequential refinement."""

    def __init__(self, channels: int, reduction: int = 16, lambd: float = 1e-4, alpha: float = 1.0) -> None:
        super().__init__()
        self.refine = SENetV2LocalVarianceSimAM(
            channels, reduction=reduction, lambd=lambd, alpha=alpha
        )
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = self.refine(x)
        return x + self.beta * (refined - x)


class _ContrastCueEncoder(nn.Module):
    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        hidden = max(8, min(hidden, channels))
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, cue: torch.Tensor) -> torch.Tensor:
        return self.net(cue)


class _ContrastStatsMixin:
    def _record_stats(self, gate: torch.Tensor) -> None:
        flat = gate.detach().float().flatten(1)
        q = torch.quantile(flat, flat.new_tensor((0.10, 0.50, 0.90)), dim=1)
        self.last_attention_stats = {
            "mean": float(flat.mean()), "std": float(flat.std()),
            "min": float(flat.min()), "max": float(flat.max()),
            "p10": float(q[0].mean()), "p50": float(q[1].mean()), "p90": float(q[2].mean()),
            "gamma": float(self.gamma.detach()),
        }


class LocalChromaContrast9x9(nn.Module):
    """Compute the deterministic full-resolution local-chroma cue once."""

    out_channels = 1

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return explicit_cue(rgb, "local_chroma_contrast", window=9)


class SENetV2EnSimAM(nn.Module):
    """Sequential high-resolution refinement shared by both ablations."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.senet = SENetV2(channels)
        self.ensimam = EnSimAM()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.ensimam(self.senet(feature))


class _PrecomputedLocalChromaGuidance(nn.Module, _ContrastStatsMixin):
    """Common per-level encoder/gate consuming a precomputed C_full tensor."""

    def __init__(self, channels: list[int] | tuple[int, int], hidden: int = 32) -> None:
        super().__init__()
        feature_ch, cue_ch = channels
        if cue_ch != 1:
            raise ValueError(f"Expected a one-channel precomputed cue, got {cue_ch}")
        self.encoder = _ContrastCueEncoder(feature_ch, hidden)
        gate_hidden = max(8, min(hidden, feature_ch))
        self.gate = nn.Sequential(
            nn.Conv2d(2 * feature_ch, gate_hidden, 1, bias=False), nn.SiLU(),
            nn.Conv2d(gate_hidden, gate_hidden, 3, 1, 1, groups=gate_hidden, bias=False), nn.SiLU(),
            nn.Conv2d(gate_hidden, 1, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(()))
        self.last_attention_stats: dict[str, float] = {}

    def _encoded_gate(self, feature: torch.Tensor, cue_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cue = F.interpolate(cue_full, size=feature.shape[-2:], mode="area")
        projected = self.encoder(cue)
        gate = torch.sigmoid(self.gate(torch.cat((feature, projected), dim=1)))
        self._record_stats(gate)
        return projected, gate


class LocalChromaSharedGuidance(_PrecomputedLocalChromaGuidance):
    """Shared residual fusion followed sequentially by SENetV2 and EnSimAM."""

    def __init__(self, channels: list[int] | tuple[int, int], hidden: int = 32) -> None:
        super().__init__(channels, hidden)
        self.refine = SENetV2EnSimAM(channels[0])

    def forward(self, values: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feature, cue_full = values
        projected, gate = self._encoded_gate(feature, cue_full)
        return self.refine(feature + self.gamma * gate * projected)


class LocalChromaClsGuidance(_PrecomputedLocalChromaGuidance):
    """Classification-only multiplicative modulation from precomputed C_full."""

    def forward(self, values: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feature, cue_full = values
        _, gate = self._encoded_gate(feature, cue_full)
        return feature * (1.0 + self.gamma * gate)


class ContrastSharedSENetV2EnSimAM(nn.Module, _ContrastStatsMixin):
    """Contrast residual guidance followed by SENetV2 and EnSimAM refinement."""

    def __init__(self, channels: list[int] | tuple[int, int], hidden: int = 32, cue: str = "contrast", window: int = 7) -> None:
        super().__init__()
        feature_ch, rgb_ch = channels
        if rgb_ch != 3:
            raise ValueError("ContrastSharedSENetV2EnSimAM requires an RGB tap")
        self.cue = cue
        if window < 3 or window % 2 == 0:
            raise ValueError(f"local contrast window must be odd and >=3, got {window}")
        self.window = window
        self.encoder = _ContrastCueEncoder(feature_ch, hidden)
        gate_hidden = max(8, min(hidden, feature_ch))
        self.gate = nn.Sequential(
            nn.Conv2d(2 * feature_ch, gate_hidden, 1, bias=False), nn.SiLU(),
            nn.Conv2d(gate_hidden, gate_hidden, 3, 1, 1, groups=gate_hidden, bias=False), nn.SiLU(),
            nn.Conv2d(gate_hidden, 1, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(()))
        self.senet = SENetV2(feature_ch)
        self.ensimam = EnSimAM()
        self.last_attention_stats: dict[str, float] = {}

    def forward(self, values: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feature, rgb = values
        cue = F.interpolate(explicit_cue(rgb, self.cue, window=self.window), size=feature.shape[-2:], mode="area")
        projected = self.encoder(cue)
        gate = torch.sigmoid(self.gate(torch.cat((feature, projected), dim=1)))
        self._record_stats(gate)
        return self.ensimam(self.senet(feature + self.gamma * gate * projected))


class ContrastClsGuidance(nn.Module, _ContrastStatsMixin):
    """Contrast-conditioned classification-only multiplicative guidance."""

    def __init__(self, channels: list[int] | tuple[int, int], hidden: int = 32, cue: str = "contrast", window: int = 7) -> None:
        super().__init__()
        feature_ch, rgb_ch = channels
        if rgb_ch != 3:
            raise ValueError("ContrastClsGuidance requires an RGB tap")
        self.cue = cue
        if window < 3 or window % 2 == 0:
            raise ValueError(f"local contrast window must be odd and >=3, got {window}")
        self.window = window
        self.encoder = _ContrastCueEncoder(feature_ch, hidden)
        gate_hidden = max(8, min(hidden, feature_ch))
        self.gate = nn.Sequential(
            nn.Conv2d(2 * feature_ch, gate_hidden, 1, bias=False), nn.SiLU(),
            nn.Conv2d(gate_hidden, gate_hidden, 3, 1, 1, groups=gate_hidden, bias=False), nn.SiLU(),
            nn.Conv2d(gate_hidden, 1, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(()))
        self.last_attention_stats: dict[str, float] = {}

    def forward(self, values: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feature, rgb = values
        cue = F.interpolate(explicit_cue(rgb, self.cue, window=self.window), size=feature.shape[-2:], mode="area")
        projected = self.encoder(cue)
        gate = torch.sigmoid(self.gate(torch.cat((feature, projected), dim=1)))
        self._record_stats(gate)
        return feature * (1.0 + self.gamma * gate)


class ASFHighResFusion(nn.Module):
    """ASF semantic enrichment for P2/P3 recipients only."""

    def __init__(self, channels: list[int] | tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"ASFHighResFusion expects detail + two semantic inputs, got {channels}")
        self.scale_fusion = ScalSeq(channels, out_channels)
        self.asf = ASFAttention(out_channels)

    def forward(self, values: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.asf(self.scale_fusion(values))


__all__ = (
    "SENetV2", "LocalVarianceSimAM", "SENetV2LocalVarianceSimAM",
    "ResidualSENetV2LocalVarianceSimAM", "SENetV2EnSimAM",
    "LocalChromaContrast9x9", "LocalChromaSharedGuidance",
    "LocalChromaClsGuidance", "ContrastSharedSENetV2EnSimAM", "ContrastClsGuidance", "ASFHighResFusion",
)
