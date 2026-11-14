# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""P2'-guided operators for controlled P3/P4 ablations.

Every block consumes ``[target, p2_guidance]``, preserves the target shape and
uses a zero-initialized residual scale so the initial mapping is identity.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = (
    "P2GuidedGatedModulation",
    "P2GuidedDynamicConv",
    "P2GuidedLocalCrossAttention",
    "P2GuidedDeformConv",
)


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _resize_guidance(guidance: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Area-downsample P2' onto the real target grid without hard-coded sizes."""
    return F.interpolate(guidance, size=target.shape[-2:], mode="area")


class P2GuidedGatedModulation(nn.Module):
    """Generate spatial/channel gates from P2' and multiplicatively modulate a target."""

    def __init__(self, channels: list[int], embed_channels: int = 32) -> None:
        super().__init__()
        target_channels, guidance_channels = channels
        self.target_proj = nn.Sequential(
            nn.Conv2d(target_channels, embed_channels, 1, bias=False),
            nn.GroupNorm(_groups(embed_channels), embed_channels),
            nn.SiLU(),
        )
        self.guidance_proj = nn.Sequential(
            nn.Conv2d(guidance_channels, embed_channels, 1, bias=False),
            nn.GroupNorm(_groups(embed_channels), embed_channels),
            nn.SiLU(),
        )
        self.spatial_gate = nn.Conv2d(embed_channels * 2, 1, 3, padding=1)
        hidden = max(target_channels // 4, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, target_channels, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(1, target_channels, 1, 1))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target, guidance = features
        guidance = _resize_guidance(guidance, target)
        target_embed = self.target_proj(target)
        guidance_embed = self.guidance_proj(guidance)
        spatial = torch.sigmoid(self.spatial_gate(torch.cat((target_embed, guidance_embed), dim=1)))
        channel = torch.sigmoid(self.channel_gate(guidance_embed))
        return target * (1.0 + self.gamma * spatial * channel)


class P2GuidedDynamicConv(nn.Module):
    """Mix a small bank of learned convolution experts using coefficients from P2'."""

    def __init__(
        self,
        channels: list[int],
        embed_channels: int = 32,
        num_experts: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        target_channels, guidance_channels = channels
        if num_experts < 2:
            raise ValueError("P2GuidedDynamicConv requires at least two experts")
        self.guidance_proj = nn.Sequential(
            nn.Conv2d(guidance_channels, embed_channels, 1, bias=False),
            nn.GroupNorm(_groups(embed_channels), embed_channels),
            nn.SiLU(),
        )
        hidden = max(embed_channels // 2, num_experts)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_experts),
        )
        padding = kernel_size // 2
        # Depthwise experts focus on spatial mixing; a shared 1x1 projection
        # performs channel mixing after the content-dependent expert mixture.
        self.experts = nn.ModuleList(
            nn.Conv2d(
                target_channels,
                target_channels,
                kernel_size,
                padding=padding,
                groups=target_channels,
                bias=False,
            )
            for _ in range(num_experts)
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(target_channels, target_channels, 1, bias=False),
            nn.GroupNorm(_groups(target_channels), target_channels),
            nn.SiLU(),
        )
        self.gamma = nn.Parameter(torch.zeros(1, target_channels, 1, 1))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target, guidance = features
        guidance = self.guidance_proj(_resize_guidance(guidance, target))
        coefficients = self.router(guidance).softmax(dim=1)
        expert_outputs = torch.stack([expert(target) for expert in self.experts], dim=1)
        mixed = (expert_outputs * coefficients[:, :, None, None, None]).sum(dim=1)
        return target + self.gamma * self.out_proj(mixed)


class P2GuidedLocalCrossAttention(nn.Module):
    """Local-window cross-attention with target queries and P2' keys/values."""

    def __init__(
        self,
        channels: list[int],
        embed_channels: int = 32,
        num_heads: int = 4,
        window_size: int = 3,
    ) -> None:
        super().__init__()
        target_channels, guidance_channels = channels
        if embed_channels % num_heads:
            raise ValueError("embed_channels must be divisible by num_heads")
        if window_size % 2 != 1:
            raise ValueError("window_size must be odd")
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.head_dim = embed_channels // num_heads
        self.window_size = window_size
        self.q_proj = nn.Conv2d(target_channels, embed_channels, 1, bias=False)
        self.k_proj = nn.Conv2d(guidance_channels, embed_channels, 1, bias=False)
        self.v_proj = nn.Conv2d(guidance_channels, embed_channels, 1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_channels, target_channels, 1, bias=False),
            nn.GroupNorm(_groups(target_channels), target_channels),
        )
        self.gamma = nn.Parameter(torch.zeros(1, target_channels, 1, 1))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target, guidance = features
        guidance = _resize_guidance(guidance, target)
        batch, _, height, width = target.shape
        neighbors = self.window_size * self.window_size

        query = self.q_proj(target).view(
            batch, self.num_heads, self.head_dim, height, width
        )
        key = F.unfold(
            self.k_proj(guidance), self.window_size, padding=self.window_size // 2
        ).view(batch, self.num_heads, self.head_dim, neighbors, height, width)
        value = F.unfold(
            self.v_proj(guidance), self.window_size, padding=self.window_size // 2
        ).view(batch, self.num_heads, self.head_dim, neighbors, height, width)
        attention = (query.unsqueeze(3) * key).sum(dim=2) / math.sqrt(self.head_dim)
        attention = attention.softmax(dim=2)
        attended = (attention.unsqueeze(2) * value).sum(dim=3)
        attended = attended.reshape(batch, self.embed_channels, height, width)
        return target + self.gamma * self.out_proj(attended)


class P2GuidedDeformConv(nn.Module):
    """DCNv2-style target sampling whose offsets and masks are generated from P2'."""

    def __init__(
        self,
        channels: list[int],
        embed_channels: int = 32,
        kernel_size: int = 3,
        use_mask: bool = True,
    ) -> None:
        super().__init__()
        target_channels, guidance_channels = channels
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.use_mask = use_mask
        self.guidance_proj = nn.Sequential(
            nn.Conv2d(guidance_channels, embed_channels, 1, bias=False),
            nn.GroupNorm(_groups(embed_channels), embed_channels),
            nn.SiLU(),
            nn.Conv2d(embed_channels, embed_channels, 3, padding=1, bias=False),
            nn.SiLU(),
        )
        points = kernel_size * kernel_size
        self.offset_gen = nn.Conv2d(embed_channels, 2 * points, 3, padding=1)
        self.mask_gen = nn.Conv2d(embed_channels, points, 3, padding=1) if use_mask else None
        self.weight = nn.Parameter(
            torch.empty(target_channels, target_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(target_channels))
        self.norm = nn.GroupNorm(_groups(target_channels), target_channels)
        self.act = nn.SiLU()
        self.gamma = nn.Parameter(torch.zeros(1, target_channels, 1, 1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.offset_gen.weight)
        nn.init.zeros_(self.offset_gen.bias)
        if self.mask_gen is not None:
            nn.init.zeros_(self.mask_gen.weight)
            nn.init.zeros_(self.mask_gen.bias)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target, guidance = features
        guidance = self.guidance_proj(_resize_guidance(guidance, target))
        offset = self.offset_gen(guidance)
        mask = torch.sigmoid(self.mask_gen(guidance)) if self.mask_gen is not None else None
        sampled = self._deform_conv(target, offset, mask)
        return target + self.gamma * self.act(self.norm(sampled))

    def _deform_conv(
        self,
        target: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pure-PyTorch modulated deformable convolution using bilinear sampling."""
        batch, _, height, width = target.shape
        dtype, device = target.dtype, target.device
        y_base, x_base = torch.meshgrid(
            torch.arange(height, dtype=dtype, device=device),
            torch.arange(width, dtype=dtype, device=device),
            indexing="ij",
        )
        output = target.new_zeros((batch, self.weight.shape[0], height, width))
        point = 0
        for kernel_y in range(self.kernel_size):
            for kernel_x in range(self.kernel_size):
                y = y_base + (kernel_y - self.padding) + offset[:, 2 * point]
                x = x_base + (kernel_x - self.padding) + offset[:, 2 * point + 1]
                y = 2.0 * y / max(height - 1, 1) - 1.0
                x = 2.0 * x / max(width - 1, 1) - 1.0
                grid = torch.stack((x, y), dim=-1)
                sampled = F.grid_sample(
                    target,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                if mask is not None:
                    sampled = sampled * mask[:, point : point + 1]
                output = output + torch.einsum(
                    "bchw,oc->bohw", sampled, self.weight[:, :, kernel_y, kernel_x]
                )
                point += 1
        return output + self.bias.view(1, -1, 1, 1)
