# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Content-Aware ReAssembly of FEatures (CARAFE) upsampling.

This is a pure-PyTorch implementation of Wang et al., ICCV 2019. It predicts a
normalized, location-specific spatial kernel and uses it to reassemble a local
neighborhood. Channels are never mixed by the reassembly operation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ("CARAFE", "P2GuidedCARAFE")


class CARAFE(nn.Module):
    """Content-aware 2D upsampling with paper-default kernel sizes.

    Args:
        channels: Input/output feature channels.
        scale: Integer spatial upsampling factor.
        compressed_channels: Channels used by the kernel prediction branch.
        encoder_kernel: Kernel size of the content encoder.
        up_kernel: Spatial reassembly kernel size.
    """

    def __init__(
        self,
        channels: int,
        scale: int = 2,
        compressed_channels: int = 64,
        encoder_kernel: int = 3,
        up_kernel: int = 5,
    ) -> None:
        super().__init__()
        if scale < 1 or encoder_kernel % 2 != 1 or up_kernel % 2 != 1:
            raise ValueError("scale must be positive and CARAFE kernels must be odd")
        self.channels = channels
        self.scale = scale
        self.up_kernel = up_kernel
        self.compressor = nn.Conv2d(channels, compressed_channels, 1)
        self.encoder = nn.Conv2d(
            compressed_channels,
            scale * scale * up_kernel * up_kernel,
            encoder_kernel,
            padding=encoder_kernel // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample ``x`` while preserving channels and dtype/device."""
        batch, channels, height, width = x.shape
        kernel_area = self.up_kernel * self.up_kernel

        # B x (s^2 k^2) x H x W -> B x k^2 x sH x sW.
        weights = F.pixel_shuffle(self.encoder(self.compressor(x)), self.scale)
        weights = weights.softmax(dim=1)

        # Gather each low-resolution k x k neighborhood, then associate the
        # same neighborhood with its s x s output sub-pixels.
        patches = F.unfold(x, self.up_kernel, padding=self.up_kernel // 2)
        patches = patches.view(batch, channels, kernel_area, height, width)
        patches = patches.repeat_interleave(self.scale, dim=3).repeat_interleave(self.scale, dim=4)
        return (patches * weights.unsqueeze(1)).sum(dim=2)


class P2GuidedCARAFE(nn.Module):
    """Upsample P3 while using downsampled P2' to predict reassembly kernels.

    Input order is ``[source_p3, guidance_p2]``. P2' is resized to the actual
    P3 spatial shape and participates only in kernel prediction; values being
    reassembled always come from P3.
    """

    def __init__(
        self,
        channels: list[int] | tuple[int, int],
        scale: int = 2,
        compressed_channels: int = 64,
        encoder_kernel: int = 3,
        up_kernel: int = 5,
    ) -> None:
        super().__init__()
        if len(channels) != 2:
            raise ValueError("P2GuidedCARAFE expects [P3 channels, P2' channels]")
        if scale < 1 or encoder_kernel % 2 != 1 or up_kernel % 2 != 1:
            raise ValueError("scale must be positive and CARAFE kernels must be odd")
        source_channels, guidance_channels = channels
        source_embed = compressed_channels // 2
        guidance_embed = compressed_channels - source_embed
        self.scale = scale
        self.up_kernel = up_kernel
        self.source_compressor = nn.Conv2d(source_channels, source_embed, 1)
        self.guidance_compressor = nn.Conv2d(guidance_channels, guidance_embed, 1)
        self.encoder = nn.Conv2d(
            compressed_channels,
            scale * scale * up_kernel * up_kernel,
            encoder_kernel,
            padding=encoder_kernel // 2,
        )

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Reassemble P3 using kernels conditioned jointly on P3 and P2'."""
        source, guidance = features
        batch, channels, height, width = source.shape
        kernel_area = self.up_kernel * self.up_kernel

        # Area downsampling makes P2' an aligned low-resolution guide without
        # injecting its values directly into the reassembled P3 feature.
        guidance = F.interpolate(guidance, size=(height, width), mode="area")
        content = torch.cat(
            (self.source_compressor(source), self.guidance_compressor(guidance)), dim=1
        )
        weights = F.pixel_shuffle(self.encoder(content), self.scale).softmax(dim=1)

        patches = F.unfold(source, self.up_kernel, padding=self.up_kernel // 2)
        patches = patches.view(batch, channels, kernel_area, height, width)
        patches = patches.repeat_interleave(self.scale, dim=3).repeat_interleave(self.scale, dim=4)
        return (patches * weights.unsqueeze(1)).sum(dim=2)
