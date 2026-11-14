# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Adaptive-SRM feature-enhancement blocks for tiny-object (LEVIR-Ship) backbones.

Three architectures share the same :class:`AdaptiveSRM` primitive:

* :class:`P2ASRMFusion` (Version 1) - SRM residual computed directly on P2, gated and
  fused back into P2. Only touches P2; P3-P5 are untouched.
* :class:`InputASRMAuxiliaryAttention` (Version 2) - SRM residual computed on the input
  image, cross-attended against P2 to produce an auxiliary attention map. The detection
  path (P2 itself) is never modified; the attention map is exposed on
  ``self.last_attention_map`` for an external auxiliary loss to consume.
* :class:`InputASRMAttentionGuidedP2` (Version 3) - same image-level SRM + attention as
  Version 2, but the attention map is used to modulate/inject into P2 directly, in both
  train and eval.

``AdaptiveSRM`` is initialized from six train-split-selected LEVIR-Ship SRM kernels
(K24, K12, K22, K18, K14, K10 of the standard 30-filter SRM bank). A separate
three-kernel ablation can explicitly request the best response from each of the
first-, second-, and third-order directional groups: K4, K12, K16. Weights start
at these values and, in
``srm_mode="adaptive"`` (default), are fine-tuned end-to-end.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .conv import ChannelAttention, SpatialAttention

__all__ = (
    "AdaptiveSRM",
    "RGBToHSV",
    "RGBColorAugment",
    "P2ASRMFusion",
    "InputASRMAuxiliaryAttention",
    "InputASRMAttentionGuidedP2",
    "ASRMStructuralPrior",
    "SPGFusion",
    "ASRMDetailPriorDownsample",
    "ASRMP4SemanticVerifyResidual",
    "ContextAwareASRMRefine",
    "ASRMCandidateSparseRefine",
)


def _group_norm_groups(channels: int) -> int:
    """Largest group count in {32,16,8,4,2,1} that evenly divides ``channels``."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


def _grayscale(image: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma from an NCHW RGB tensor in [0, 1]."""
    weights = image.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    return (image * weights).sum(1, keepdim=True)


class RGBToHSV(nn.Module):
    """Differentiable RGB-to-HSV transform for an NCHW image in ``[0, 1]``.

    The output remains three-channel ``[H, S, V]`` in ``[0, 1]``. It is a
    parameter-free input transform, so a detector can use HSV throughout its
    backbone while preserving the normal 3-channel interface.
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"RGBToHSV expects [B,3,H,W], got {tuple(image.shape)}")
        red, green, blue = image.unbind(dim=1)
        maximum, max_index = image.max(dim=1)
        minimum = image.min(dim=1).values
        delta = maximum - minimum
        nonzero = delta > torch.finfo(image.dtype).eps
        safe_delta = delta.clamp_min(torch.finfo(image.dtype).eps)

        hue_red = torch.remainder((green - blue) / safe_delta, 6.0)
        hue_green = (blue - red) / safe_delta + 2.0
        hue_blue = (red - green) / safe_delta + 4.0
        hue = torch.where(max_index == 0, hue_red, torch.where(max_index == 1, hue_green, hue_blue)) / 6.0
        hue = torch.where(nonzero, hue, torch.zeros_like(hue))
        saturation = torch.where(maximum > 0, delta / maximum.clamp_min(torch.finfo(image.dtype).eps), torch.zeros_like(delta))
        return torch.stack((hue, saturation, maximum), dim=1)


class RGBColorAugment(nn.Module):
    """Append parameter-free colour cues to RGB while retaining the original RGB.

    ``append_saturation`` appends HSV saturation. ``append_labb`` appends the
    CIE Lab *b* component, normalized from its conventional approximately
    ``[-128, 127]`` range to ``[0, 1]``. Thus this module emits 4 channels for
    either single-cue ablation and 5 channels when both cues are enabled.
    """

    def __init__(self, append_saturation: bool = False, append_labb: bool = False) -> None:
        super().__init__()
        if not (append_saturation or append_labb):
            raise ValueError("RGBColorAugment needs saturation and/or Lab-b enabled")
        self.append_saturation = append_saturation
        self.append_labb = append_labb

    @property
    def out_channels(self) -> int:
        return 3 + int(self.append_saturation) + int(self.append_labb)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"RGBColorAugment expects [B,3,H,W], got {tuple(image.shape)}")
        red, green, blue = image.unbind(dim=1)
        # Equivalent RGB extrema using elementwise ops rather than channel-axis
        # ReduceMax/ReduceMin. On Jetson TensorRT 10.3 the FP32 reduction tactic
        # for these 3-channel tensors can dominate the entire engine latency.
        maximum = torch.maximum(torch.maximum(red, green), blue)
        minimum = torch.minimum(torch.minimum(red, green), blue)
        cues = [image]
        if self.append_saturation:
            saturation = torch.where(maximum > 0, (maximum - minimum) / maximum.clamp_min(torch.finfo(image.dtype).eps), torch.zeros_like(maximum))
            cues.append(saturation.unsqueeze(1))
        if self.append_labb:
            linear = torch.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055).pow(2.4))
            xyz = torch.einsum(
                "bchw,dc->bdhw", linear,
                image.new_tensor(((0.4124564, 0.3575761, 0.1804375), (0.2126729, 0.7151522, 0.0721750), (0.0193339, 0.1191920, 0.9503041))),
            )
            xyz = xyz / image.new_tensor((0.95047, 1.0, 1.08883)).view(1, 3, 1, 1)
            delta = 6 / 29
            fxyz = torch.where(xyz > delta**3, xyz.clamp_min(0).pow(1 / 3), xyz / (3 * delta**2) + 4 / 29)
            lab_b = 200 * (fxyz[:, 1] - fxyz[:, 2])
            cues.append(((lab_b + 128.0) / 255.0).clamp(0.0, 1.0).unsqueeze(1))
        return torch.cat(cues, dim=1)


class AdaptiveSRM(nn.Module):
    """5x5 high-pass residual filter bank initialized from six selected SRM kernels.

    K24/K12/K22/K18/K14/K10 are the manually selected subset from the train-only
    relevance and redundancy pipeline in ``SRM_kernel_selection.ipynb``.

    Two usage patterns, selected by ``depthwise``:

    * ``depthwise=True``: every input channel is filtered by one kernel from the bank
      (cycled), output channel count equals input channel count (``groups=in_channels``).
      Used to filter a multi-channel feature map such as P2 in place.
    * ``depthwise=False``: a small number of input channels (typically 1 grayscale
      channel) produce ``out_channels`` residual maps, one per kernel
      (``groups=1``). Used to turn a grayscale image into 3 residual maps.
    """

    _BASE_KERNELS = (
        ((0,0,0,0,0),(0,0,0,0,0),(0,2,-4,2,0),(0,-1,2,-1,0),(0,0,0,0,0)),  # K24
        ((0,0,0,0,0),(0,1,0,0,0),(0,0,-2,0,0),(0,0,0,1,0),(0,0,0,0,0)),  # K12
        ((0,0,0,0,0),(0,-1,2,-1,0),(0,2,-4,2,0),(0,0,0,0,0),(0,0,0,0,0)),  # K22
        ((0,0,0,0,0),(0,0,0,1,0),(0,0,-3,0,0),(0,3,0,0,0),(-1,0,0,0,0)),  # K18
        ((0,0,0,0,-1),(0,0,0,3,0),(0,0,-3,0,0),(0,1,0,0,0),(0,0,0,0,0)),  # K14
        ((0,0,0,0,0),(0,0,0,1,0),(0,0,-2,0,0),(0,1,0,0,0),(0,0,0,0,0)),  # K10
    )
    # Train-only, per-family winners from the canonical 30-kernel bank. These
    # are deliberately an opt-in bank: existing experiments retain _BASE_KERNELS.
    _TOP_DIRECTIONAL_KERNEL_IDS = (4, 12, 16)
    _TOP_DIRECTIONAL_KERNELS = (
        ((0,0,0,0,0),(0,0,0,0,0),(0,0,-1,0,0),(0,0,0,1,0),(0,0,0,0,0)),  # K4
        ((0,0,0,0,0),(0,1,0,0,0),(0,0,-2,0,0),(0,0,0,1,0),(0,0,0,0,0)),  # K12
        ((0,0,0,0,0),(0,1,0,0,0),(0,0,-3,0,0),(0,0,0,3,0),(0,0,0,0,-1)),  # K16
    )

    def __init__(self, in_channels: int, out_channels: int | None = None,
                 srm_mode: str = "adaptive", depthwise: bool = False,
                 kernel_ids: tuple[int, ...] | list[int] | None = None) -> None:
        """Initialize AdaptiveSRM.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int | None): Output channels for the ``depthwise=False`` case;
                defaults to the kernel-bank size (3). Ignored (must equal
                ``in_channels`` or be ``None``) when ``depthwise=True``.
            srm_mode (str): ``"adaptive"`` keeps the SRM-initialized weights trainable;
                ``"fixed"`` freezes them at the SRM values.
            depthwise (bool): See class docstring.
            kernel_ids: Optional explicitly supported SRM subset. ``[4, 12, 16]``
                selects the train-only per-directional-family winners. ``None``
                preserves the historical six-kernel bank exactly.
        """
        super().__init__()
        if srm_mode not in ("adaptive", "fixed"):
            raise ValueError(f"srm_mode must be 'adaptive' or 'fixed', got {srm_mode!r}")
        if kernel_ids is None:
            selected_ids = None
            selected_kernels = self._BASE_KERNELS
        else:
            selected_ids = tuple(int(kernel_id) for kernel_id in kernel_ids)
            if selected_ids != self._TOP_DIRECTIONAL_KERNEL_IDS:
                raise ValueError(
                    "AdaptiveSRM currently supports explicit kernel_ids=(4, 12, 16) only; "
                    f"got {selected_ids}"
                )
            selected_kernels = self._TOP_DIRECTIONAL_KERNELS
        bank = torch.tensor(selected_kernels, dtype=torch.float32).unsqueeze(1)  # [K,1,5,5]
        n_kernels = bank.shape[0]
        if depthwise:
            if out_channels not in (None, in_channels):
                raise ValueError("depthwise AdaptiveSRM requires out_channels == in_channels")
            out_channels = in_channels
            groups = in_channels
            weight = bank[[i % n_kernels for i in range(in_channels)]]  # [C,1,5,5]
        else:
            out_channels = out_channels or n_kernels
            groups = 1
            reps = math.ceil(out_channels / n_kernels)
            weight = bank.repeat(reps, 1, 1, 1)[:out_channels]  # [out,1,5,5]
            weight = weight.repeat(1, in_channels, 1, 1) / in_channels  # [out,in,5,5]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2,
                               groups=groups, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        self.conv.weight.requires_grad_(srm_mode == "adaptive")
        self.out_channels = out_channels
        self.kernel_ids = selected_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _GatedFusion(nn.Module):
    """``sigmoid(Conv1x1(concat(feature, residual)))``, shaped by ``gate_type``."""

    def __init__(self, channels: int, gate_type: str = "spatial_channel") -> None:
        super().__init__()
        if gate_type not in ("spatial", "channel", "spatial_channel"):
            raise ValueError(f"gate_type must be spatial/channel/spatial_channel, got {gate_type!r}")
        self.gate_type = gate_type
        if gate_type in ("spatial", "spatial_channel"):
            out = 1 if gate_type == "spatial" else channels
            self.spatial_gate = nn.Conv2d(channels * 2, out, 1)
        if gate_type in ("channel", "spatial_channel"):
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.channel_gate = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, feature: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        cat = torch.cat((feature, residual), dim=1)
        if self.gate_type == "spatial":
            return torch.sigmoid(self.spatial_gate(cat))
        if self.gate_type == "channel":
            return torch.sigmoid(self.channel_gate(self.pool(cat)))
        return torch.sigmoid(self.spatial_gate(cat) + self.channel_gate(self.pool(cat)))


class P2ASRMFusion(nn.Module):
    """Version 1: Adaptive SRM residual on P2, gated fusion back into P2.

    ``P2_out = P2 + gamma * gate * R2``, ``R2`` a projected SRM residual, ``gamma``
    zero-initialized so the module starts as an identity. Meant to be inserted as a tap
    right after the P2 backbone stage (see ``model_cfg/yolov8_p2_asrm_fusion.yaml``);
    P3-P5 are never touched.
    """

    def __init__(self, channels: int, srm_mode: str = "adaptive",
                 gate_type: str = "spatial_channel", gamma_init: float = 0.0,
                 gamma_channel_wise: bool = True, debug: bool = False) -> None:
        """Initialize P2ASRMFusion.

        Args:
            channels (int): P2 channel count (inferred automatically from the graph
                when built through ``parse_model``/YAML).
            srm_mode (str): ``"adaptive"`` or ``"fixed"``, see :class:`AdaptiveSRM`.
            gate_type (str): ``"spatial"``, ``"channel"`` or ``"spatial_channel"``.
            gamma_init (float): Initial value of the learnable fusion gain.
            gamma_channel_wise (bool): Channel-wise ``gamma`` (default) vs a single
                scalar shared across channels.
            debug (bool): When True, ``forward`` also returns the gate map.
        """
        super().__init__()
        self.debug = debug
        self.srm = AdaptiveSRM(channels, srm_mode=srm_mode, depthwise=True)
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(_group_norm_groups(channels), channels),
            nn.SiLU(inplace=True),
        )
        self.gate = _GatedFusion(channels, gate_type=gate_type)
        shape = (1, channels, 1, 1) if gamma_channel_wise else ()
        self.gamma = nn.Parameter(torch.full(shape, float(gamma_init)))

    def forward(self, p2: torch.Tensor):
        residual = self.project(self.srm(p2))
        gate_map = self.gate(p2, residual)
        out = p2 + self.gamma * gate_map * residual
        return (out, gate_map) if self.debug else out


class WindowCrossAttention(nn.Module):
    """Lightweight non-overlapping local-window cross attention (Q from one source, K/V
    from another), used instead of full H*W global attention to keep memory bounded.
    """

    def __init__(self, embed_channels: int, window_size: int = 8, num_heads: int = 4) -> None:
        super().__init__()
        if embed_channels % num_heads != 0:
            raise ValueError(f"embed_channels ({embed_channels}) must be divisible by num_heads ({num_heads})")
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = embed_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Conv2d(embed_channels, embed_channels, 1)
        self.k = nn.Conv2d(embed_channels, embed_channels, 1)
        self.v = nn.Conv2d(embed_channels, embed_channels, 1)
        self.proj = nn.Conv2d(embed_channels, embed_channels, 1)

    def _windows(self, t: torch.Tensor, b: int, hp: int, wp: int) -> torch.Tensor:
        ws = self.window_size
        t = t.view(b, self.num_heads, self.head_dim, hp // ws, ws, wp // ws, ws)
        t = t.permute(0, 3, 5, 1, 4, 6, 2)  # b, nH, nW, heads, ws, ws, head_dim
        return t.reshape(b * (hp // ws) * (wp // ws), self.num_heads, ws * ws, self.head_dim)

    def forward(self, query_src: torch.Tensor, kv_src: torch.Tensor) -> torch.Tensor:
        b, c, h, w = query_src.shape
        ws = self.window_size
        pad_h, pad_w = (ws - h % ws) % ws, (ws - w % ws) % ws
        q_full = F.pad(self.q(query_src), (0, pad_w, 0, pad_h))
        k_full = F.pad(self.k(kv_src), (0, pad_w, 0, pad_h))
        v_full = F.pad(self.v(kv_src), (0, pad_w, 0, pad_h))
        hp, wp = q_full.shape[-2:]
        qw, kw, vw = self._windows(q_full, b, hp, wp), self._windows(k_full, b, hp, wp), self._windows(v_full, b, hp, wp)
        # SRM responses can be considerably larger than ordinary image features.
        # Compute the logits in fp32 and bound them before softmax: under AMP, an
        # fp16 matmul may otherwise overflow and poison the entire attention map.
        logits = qw.float() @ kw.float().transpose(-2, -1)
        logits = torch.nan_to_num(logits * self.scale, nan=0.0, posinf=50.0, neginf=-50.0)
        attn = logits.clamp_(-50.0, 50.0).softmax(dim=-1).to(dtype=vw.dtype)
        out = attn @ vw
        n_h, n_w = hp // ws, wp // ws
        out = out.reshape(b, n_h, n_w, self.num_heads, ws, ws, self.head_dim)
        out = out.permute(0, 3, 6, 1, 4, 2, 5).reshape(b, c, hp, wp)
        return self.proj(out)[:, :, :h, :w]


class _CBAMCrossAttention(nn.Module):
    """CBAM-style channel+spatial attention over concat(P2 embedding, SRM residual embedding).

    Reuses Ultralytics' own :class:`ChannelAttention`/:class:`SpatialAttention`
    (``ultralytics/nn/modules/conv.py``) instead of a bespoke implementation.
    """

    def __init__(self, embed_channels: int) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(embed_channels * 2)
        self.spatial_attention = SpatialAttention(7)
        self.to_map = nn.Conv2d(embed_channels * 2, 1, 1)

    def forward(self, p2_embed: torch.Tensor, residual_embed: torch.Tensor) -> torch.Tensor:
        cat = torch.cat((p2_embed, residual_embed), dim=1)
        cat = self.spatial_attention(self.channel_attention(cat))
        return self.to_map(cat)


class _InputASRMAttentionBase(nn.Module):
    """Shared image-level SRM + P2 cross-attention pipeline for Versions 2 and 3.

    Input: ``x = [p2, image]`` (a 2-element list, gathered by Ultralytics' generic
    multi-source ``from`` mechanism from ``f: [p2_layer_index, 0]`` in YAML; layer 0
    is an ``nn.Identity`` that re-exposes the raw model input, see the ``model_cfg/``
    YAML files for the exact wiring).
    """

    def __init__(self, p2_channels: int, embed_channels: int, srm_mode: str,
                 attention_type: str, window_size: int,
                 kernel_ids: tuple[int, ...] | list[int] | None = None) -> None:
        super().__init__()
        if attention_type not in ("window", "cbam"):
            raise ValueError(f"attention_type must be 'window' or 'cbam', got {attention_type!r}")
        self.attention_type = attention_type
        residual_channels = len(kernel_ids) if kernel_ids is not None else 6
        self.srm = AdaptiveSRM(
            1, out_channels=residual_channels, srm_mode=srm_mode,
            depthwise=False, kernel_ids=kernel_ids,
        )
        self.residual_project = nn.Sequential(
            nn.Conv2d(residual_channels, embed_channels, 3, padding=1),
            nn.GroupNorm(_group_norm_groups(embed_channels), embed_channels),
            nn.SiLU(inplace=True),
        )
        self.p2_project = nn.Conv2d(p2_channels, embed_channels, 1)
        if attention_type == "window":
            self.attn = WindowCrossAttention(embed_channels, window_size=window_size)
            self.to_map = nn.Conv2d(embed_channels, 1, 1)
        else:
            self.attn = _CBAMCrossAttention(embed_channels)

    def _attend(self, p2: torch.Tensor, image: torch.Tensor):
        """Returns (attention_map [B,1,h,w], attention_feature [B,C,h,w] | None, residual_embed)."""
        # Keep the high-pass branch bounded before its projected/attention path.
        # This does not alter the structural-prior topology; it only prevents a
        # rare AMP overflow from turning a single augmented batch into NaNs.
        residual = self.srm(_grayscale(image))
        residual = torch.nan_to_num(residual, nan=0.0, posinf=32.0, neginf=-32.0).clamp_(-32.0, 32.0)
        residual = F.interpolate(residual, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        residual_embed = self.residual_project(residual)
        p2_embed = self.p2_project(p2)
        if self.attention_type == "window":
            feat = self.attn(p2_embed, residual_embed)
            return torch.nan_to_num(torch.sigmoid(self.to_map(feat)), nan=0.5), feat, residual_embed
        return torch.nan_to_num(torch.sigmoid(self.attn(p2_embed, residual_embed)), nan=0.5), None, residual_embed


class ASRMStructuralPrior(_InputASRMAttentionBase):
    """Generate an ASRM-guided structural prior without altering backbone features.

    Input order is ``[P2, raw_image]``. The shared AdaptiveSRM and image-guided
    attention path returns only ``A2`` with shape ``[B, 1, H2, W2]``. No SRM
    residual and no refined P2/P3 tensor is emitted.
    """

    def __init__(
        self,
        in_channels: list[int],
        embed_channels: int = 32,
        srm_mode: str = "adaptive",
        attention_type: str = "window",
        window_size: int = 8,
    ) -> None:
        if len(in_channels) != 2:
            raise ValueError("Expected [P2, raw_image] channels")
        p2_channels, image_channels = in_channels
        if image_channels != 3:
            raise ValueError(f"Raw image tap must have 3 channels, got {image_channels}")
        super().__init__(p2_channels, embed_channels, srm_mode, attention_type, window_size)
        self.last_prior_shape: tuple[int, ...] | None = None

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        p2, image = inputs
        prior, _, _ = self._attend(p2, image)
        self.last_prior_shape = tuple(prior.shape)
        return prior


class SPGFusion(nn.Module):
    """Structural-prior-guided detail/semantic fusion at a required output scale.

    ``alpha`` is a one-channel spatial gate. The prior influences only this gate:
    no raw ASRM residual is fused into either input feature.
    """

    def __init__(self, in_channels: list[int], out_channels: int) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("Expected [detail, semantic, structural_prior] channels")
        detail_c, semantic_c, prior_c = in_channels
        if prior_c != 1:
            raise ValueError(f"Structural prior must have one channel, got {prior_c}")
        self.out_channels = int(out_channels)

        def project(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, self.out_channels, 1, bias=False),
                nn.GroupNorm(_group_norm_groups(self.out_channels), self.out_channels),
                nn.SiLU(inplace=True),
            )

        self.detail_project = project(detail_c)
        self.semantic_project = project(semantic_c)
        self.prior_project = project(prior_c)
        self.gate = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(self.out_channels), self.out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.out_channels, 1, 1),
        )
        self.output_project = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(self.out_channels), self.out_channels),
            nn.SiLU(inplace=True),
        )
        self.last_detail_shape: tuple[int, ...] | None = None
        self.last_semantic_shape: tuple[int, ...] | None = None
        self.last_prior_shape: tuple[int, ...] | None = None
        self.last_output_shape: tuple[int, ...] | None = None

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        detail, semantic, prior = inputs
        size = detail.shape[-2:]
        detail_feature = self.detail_project(detail)
        semantic_feature = F.interpolate(self.semantic_project(semantic), size=size, mode="bilinear", align_corners=False)
        prior_feature = F.interpolate(self.prior_project(prior), size=size, mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.gate(detail_feature + semantic_feature + prior_feature))
        output = self.output_project(alpha * detail_feature + (1.0 - alpha) * semantic_feature)
        self.last_detail_shape = tuple(detail.shape)
        self.last_semantic_shape = tuple(semantic.shape)
        self.last_prior_shape = tuple(prior.shape)
        self.last_output_shape = tuple(output.shape)
        return output


class ASRMDetailPriorDownsample(nn.Module):
    """Learned, detail-preserving 2x downsampling from A2 to A3.

    This is prepared solely for the progressive SPG ablation. It consumes a
    one-channel prior and outputs a one-channel prior; no backbone feature is
    changed or injected.
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("ASRMDetailPriorDownsample expects a one-channel prior")
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(8), 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, prior: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.downsample(prior))


class ASRMP4SemanticVerifyResidual(nn.Module):
    """Apply a zero-initialized, P4-verified ASRM correction to final FPN P2.

    Inputs are ``[F2_base, raw_P2, raw_P4, A2]``. ASRM contributes only the
    one-channel candidate prior ``A2``; semantic content is projected from P4.
    Therefore the original FPN P2 is exactly preserved at initialization.
    """

    def __init__(self, in_channels: list[int], embed_channels: int = 32) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("Expected [F2_base, raw_P2, raw_P4, A2] channels")
        base_c, p2_c, p4_c, prior_c = in_channels
        if prior_c != 1:
            raise ValueError(f"A2 must be one channel, got {prior_c}")
        embed_channels = int(embed_channels)

        def embed(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, embed_channels, 1, bias=False),
                nn.GroupNorm(_group_norm_groups(embed_channels), embed_channels),
                nn.SiLU(inplace=True),
            )

        self.detail_project = embed(p2_c)
        self.semantic_project = embed(p4_c)
        self.verify = nn.Sequential(
            nn.Conv2d(embed_channels, embed_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(embed_channels), embed_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_channels, 1, 1),
        )
        self.correction_project = nn.Sequential(
            nn.Conv2d(embed_channels, base_c, 3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(base_c), base_c),
            nn.SiLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_shapes: dict[str, tuple[int, ...]] = {}
        self.last_gamma: float = 0.0

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        base_f2, raw_p2, raw_p4, prior_a2 = inputs
        size = base_f2.shape[-2:]
        detail = self.detail_project(raw_p2)
        semantic = F.interpolate(self.semantic_project(raw_p4), size=size, mode="bilinear", align_corners=False)

        # Raw elementwise multiplication can overflow in fp16 once either
        # projected feature has a transient large activation.  The verification
        # signal only needs agreement, so use a bounded, normalized interaction.
        detail_unit = F.normalize(torch.nan_to_num(detail.float()), dim=1, eps=1e-6)
        semantic_unit = F.normalize(torch.nan_to_num(semantic.float()), dim=1, eps=1e-6)
        interaction = (detail_unit * semantic_unit).to(dtype=detail.dtype)
        verification = torch.nan_to_num(torch.sigmoid(self.verify(interaction)), nan=0.5).clamp_(0.0, 1.0)
        prior = F.interpolate(prior_a2, size=size, mode="bilinear", align_corners=False)
        prior = torch.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        verified = prior * verification
        correction = torch.tanh(torch.nan_to_num(self.correction_project(semantic)))
        # Bound the learned residual gain without changing its zero-init identity.
        output = base_f2 + torch.tanh(self.gamma) * verified * correction
        self.last_shapes = {
            "A2": tuple(prior.shape), "D2": tuple(detail.shape), "S4": tuple(semantic.shape),
            "V2": tuple(verification.shape), "M2": tuple(verified.shape), "C2": tuple(correction.shape),
            "F2_base": tuple(base_f2.shape), "F2_out": tuple(output.shape),
        }
        self.last_gamma = float(self.gamma.detach())
        return output


class InputASRMAuxiliaryAttention(_InputASRMAttentionBase):
    """Version 2: image-level SRM auxiliary attention branch (DRE-Net-style).

    Does **not** modify P2 or the detection path: ``forward`` always returns ``p2``
    unchanged. The attention map (and, in ``window`` mode, the pre-sigmoid attention
    feature) are cached on ``self.last_attention_map`` / ``self.last_attention_feature``
    for an external training loop to read after the forward pass and use in an
    auxiliary loss; this module does not compute or register any loss itself.

    Set ``enable_auxiliary_in_eval=False`` (default) to skip the branch entirely in
    ``model.eval()`` so inference cost/behaviour matches the plain baseline.
    """

    def __init__(self, p2_channels: int, embed_channels: int = 64, srm_mode: str = "adaptive",
                 attention_type: str = "window", window_size: int = 8,
                 enable_auxiliary_in_eval: bool = False) -> None:
        super().__init__(p2_channels, embed_channels, srm_mode, attention_type, window_size)
        self.enable_auxiliary_in_eval = enable_auxiliary_in_eval
        self.last_attention_map = None
        self.last_attention_feature = None

    def forward(self, x):
        p2, image = x
        if self.training or self.enable_auxiliary_in_eval:
            attention_map, attention_feature, _ = self._attend(p2, image)
            self.last_attention_map = attention_map
            self.last_attention_feature = attention_feature
        else:
            self.last_attention_map = None
            self.last_attention_feature = None
        return p2


class InputASRMAttentionGuidedP2(_InputASRMAttentionBase):
    """Version 3: image-level SRM attention used to guide/enhance P2 directly.

    Unlike Version 2, the attention branch runs in both train and eval and its output
    feeds back into P2:

    * ``fusion_mode="attention_only"``:  ``P2_out = P2 * (1 + gamma * A2)``
    * ``fusion_mode="residual_injection"`` (default): ``P2_out = P2 + gamma * A2 * R2``

    ``gamma`` is a learnable, zero-initialized, channel-broadcast gain so the module
    starts as an identity.
    """

    def __init__(self, p2_channels: int, embed_channels: int = 64, srm_mode: str = "adaptive",
                 attention_type: str = "window", fusion_mode: str = "residual_injection",
                 gamma_init: float = 0.0, window_size: int = 8, debug: bool = False) -> None:
        super().__init__(p2_channels, embed_channels, srm_mode, attention_type, window_size)
        if fusion_mode not in ("attention_only", "residual_injection"):
            raise ValueError(f"fusion_mode must be attention_only/residual_injection, got {fusion_mode!r}")
        self.fusion_mode = fusion_mode
        self.debug = debug
        self.residual_out = nn.Conv2d(embed_channels, p2_channels, 1)
        self.gamma = nn.Parameter(torch.full((1, p2_channels, 1, 1), float(gamma_init)))

    def forward(self, x):
        p2, image = x
        attention_map, _, residual_embed = self._attend(p2, image)
        if self.fusion_mode == "attention_only":
            out = p2 * (1 + self.gamma * attention_map)
        else:
            out = p2 + self.gamma * attention_map * self.residual_out(residual_embed)
        return (out, attention_map) if self.debug else out


class ContextAwareASRMRefine(nn.Module):
    """Context-aware ASRM gate that refines one shallow feature.

    Input order is [target, peer_detail, p4, p5, image]. ASRM is used only to
    locate high-frequency evidence. Its residual is never injected into the output:
    output = target * (1 + gamma * sigmoid(gate_features)).
    """

    def __init__(
        self,
        in_channels: list[int],
        embed_channels: int = 32,
        srm_mode: str = "adaptive",
        gamma_init: float = 0.0,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if len(in_channels) != 5:
            raise ValueError("Expected channels for [target, peer, P4, P5, image]")
        target_c, peer_c, p4_c, p5_c, image_c = in_channels
        if image_c != 3:
            raise ValueError(f"Raw image tap must have 3 channels, got {image_c}")
        self.debug = debug
        self.srm = AdaptiveSRM(1, out_channels=3, srm_mode=srm_mode, depthwise=False)

        def project(channels: int, kernel_size: int = 1) -> nn.Sequential:
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv2d(channels, embed_channels, kernel_size, padding=padding, bias=False),
                nn.GroupNorm(_group_norm_groups(embed_channels), embed_channels),
                nn.SiLU(inplace=True),
            )

        self.srm_project = project(3, 3)
        self.target_project = project(target_c)
        self.peer_project = project(peer_c)
        self.p4_project = project(p4_c)
        self.p5_project = project(p5_c)
        fused_channels = embed_channels * 5
        self.context_aware_gate = nn.Sequential(
            nn.Conv2d(fused_channels, embed_channels * 2, 3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(embed_channels * 2), embed_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_channels * 2, target_c, 1),
        )
        self.gamma = nn.Parameter(torch.full((1, target_c, 1, 1), float(gamma_init)))
        self.last_gate = None

    @staticmethod
    def _resize(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        target, peer, p4, p5, image = x
        size = target.shape[-2:]
        high_frequency = self.srm(_grayscale(image)).abs()
        sources = (
            self.target_project(target),
            self._resize(self.peer_project(peer), size),
            self._resize(self.srm_project(high_frequency), size),
            self._resize(self.p4_project(p4), size),
            self._resize(self.p5_project(p5), size),
        )
        gate = torch.sigmoid(self.context_aware_gate(torch.cat(sources, dim=1)))
        output = target * (1.0 + self.gamma * gate)
        # Keep diagnostics outside the autograd graph. Ultralytics constructs
        # ModelEMA with deepcopy(model) after its stride-inference forward pass;
        # caching a non-leaf tensor here makes that deepcopy fail on PyTorch 2.11.
        self.last_gate = gate.detach()
        return (output, gate) if self.debug else output


class InputGuidedContextRefine(nn.Module):
    """Use only image-guided P2 attention to contextually refine one shallow level.

    Input order is ``[target, peer, guidance_p2, p4, p5, image]``. The image and
    P2 create a single guided prior A2; there is deliberately no image-guided P3
    branch. A2 is verified with P2/P3 detail and P4/P5 context, then the original
    target is multiplicatively refined exactly once. Raw ASRM residual is never
    injected into the output.
    """

    def __init__(
        self,
        in_channels: list[int],
        embed_channels: int = 32,
        srm_mode: str = "adaptive",
        attention_type: str = "window",
        window_size: int = 8,
        gamma_init: float = 0.0,
        debug: bool = False,
        kernel_ids: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        super().__init__()
        if len(in_channels) != 6:
            raise ValueError("Expected [target, peer, guidance_P2, P4, P5, image]")
        target_c, peer_c, guidance_c, p4_c, p5_c, image_c = in_channels
        if image_c != 3:
            raise ValueError(f"Raw image tap must have 3 channels, got {image_c}")
        self.debug = debug
        self.guidance = _InputASRMAttentionBase(
            guidance_c, embed_channels, srm_mode, attention_type, window_size, kernel_ids
        )

        def project(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, embed_channels, 1, bias=False),
                nn.GroupNorm(_group_norm_groups(embed_channels), embed_channels),
                nn.SiLU(inplace=True),
            )

        self.target_project = project(target_c)
        self.peer_project = project(peer_c)
        self.p4_project = project(p4_c)
        self.p5_project = project(p5_c)
        self.attention_project = project(1)
        self.context_gate = nn.Sequential(
            nn.Conv2d(embed_channels * 5, embed_channels * 2, 3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(embed_channels * 2), embed_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_channels * 2, target_c, 1),
        )
        self.gamma = nn.Parameter(torch.full((1, target_c, 1, 1), float(gamma_init)))
        self.last_guided_attention = None
        self.last_gate = None

    @staticmethod
    def _resize(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)

    def forward(self, inputs):
        target, peer, guidance_p2, p4, p5, image = inputs
        size = target.shape[-2:]
        attention_p2, _, _ = self.guidance._attend(guidance_p2, image)
        sources = (
            self.target_project(target),
            self._resize(self.peer_project(peer), size),
            self._resize(self.attention_project(attention_p2), size),
            self._resize(self.p4_project(p4), size),
            self._resize(self.p5_project(p5), size),
        )
        gate = torch.sigmoid(self.context_gate(torch.cat(sources, dim=1)))
        output = target * (1.0 + self.gamma * gate)
        self.last_guided_attention = attention_p2.detach()
        self.last_gate = gate.detach()
        return (output, gate, attention_p2) if self.debug else output


class InputGuidedP2P3ContextRefine(InputGuidedContextRefine):
    """Context refiner whose guidance input is the same level as its target.

    Instantiate once as ``[P2, P3, P2, P4, P5, image]`` and once as
    ``[P3, P2, P3, P4, P5, image]`` to obtain independent A2/A3 priors. The
    inherited fusion remains multiplicative and never injects raw SRM residual.
    """


class InputGuidedP34ContextRefine(nn.Module):
    """Refine P2 from image guidance plus P3/P4 background context only.

    Input order is ``[P2_target, P3_context, P4_context, image]``.  P2 keeps
    all high-resolution detail and is the only feature being transformed.  P3
    and P4 are projected and resized solely to verify whether a candidate is
    compatible with its surrounding background; P5 is intentionally absent.
    The zero-initialized residual scale makes this an exact P2 identity at
    initialization.
    """

    def __init__(
        self,
        in_channels: list[int],
        embed_channels: int = 32,
        srm_mode: str = "adaptive",
        attention_type: str = "window",
        window_size: int = 8,
        gamma_init: float = 0.0,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("Expected [P2_target, P3_context, P4_context, image]")
        target_c, p3_c, p4_c, image_c = in_channels
        if image_c != 3:
            raise ValueError(f"Raw image tap must have 3 channels, got {image_c}")
        self.debug = debug
        self.guidance = _InputASRMAttentionBase(
            target_c, embed_channels, srm_mode, attention_type, window_size
        )

        def project(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, embed_channels, 1, bias=False),
                nn.GroupNorm(_group_norm_groups(embed_channels), embed_channels),
                nn.SiLU(inplace=True),
            )

        self.target_project = project(target_c)
        self.p3_project = project(p3_c)
        self.p4_project = project(p4_c)
        self.attention_project = project(1)
        self.context_gate = nn.Sequential(
            nn.Conv2d(embed_channels * 4, embed_channels * 2, 3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(embed_channels * 2), embed_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_channels * 2, target_c, 1),
        )
        self.gamma = nn.Parameter(torch.full((1, target_c, 1, 1), float(gamma_init)))
        self.last_guided_attention = None
        self.last_gate = None

    @staticmethod
    def _resize(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)

    def forward(self, inputs):
        target, p3, p4, image = inputs
        size = target.shape[-2:]
        attention_p2, _, _ = self.guidance._attend(target, image)
        sources = (
            self.target_project(target),
            self._resize(self.attention_project(attention_p2), size),
            self._resize(self.p3_project(p3), size),
            self._resize(self.p4_project(p4), size),
        )
        gate = torch.sigmoid(self.context_gate(torch.cat(sources, dim=1)))
        output = target * (1.0 + self.gamma * gate)
        self.last_guided_attention = attention_p2.detach()
        self.last_gate = gate.detach()
        return (output, gate, attention_p2) if self.debug else output


class ASRMCandidateSparseRefine(nn.Module):
    """Verify sparse ASRM candidates against detail and background prototypes.

    Input order is ``[target, peer_detail, p4, p5, image]``. Absolute K4/K5/K8
    responses only propose candidate coordinates and provide verifier evidence; they
    are never injected into the output. P4/P5 dynamically form background prototypes,
    while P2/P3 provide candidate detail. The verified local mask refines transformed
    target features only::

        output = target + gamma * sparse_mask * phi(target)

    Candidate coordinate selection is intentionally non-differentiable (local maxima
    and top-k), while confidence verification, prototype learning, mask amplitudes and
    feature refinement remain differentiable.
    """

    def __init__(
        self,
        in_channels: list[int],
        embed_channels: int = 32,
        num_prototypes: int = 8,
        num_candidates: int = 96,
        candidate_threshold: float = 0.25,
        gaussian_sigma: float = 1.5,
        srm_mode: str = "adaptive",
        gamma_init: float = 0.0,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if len(in_channels) != 5:
            raise ValueError("Expected channels for [target, peer, P4, P5, image]")
        target_c, peer_c, p4_c, p5_c, image_c = in_channels
        if image_c != 3:
            raise ValueError(f"Raw image tap must have 3 channels, got {image_c}")
        if num_prototypes < 1 or num_candidates < 1:
            raise ValueError("num_prototypes and num_candidates must be positive")
        if gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")

        self.num_candidates = int(num_candidates)
        self.candidate_threshold = float(candidate_threshold)
        self.gaussian_sigma = float(gaussian_sigma)
        self.debug = debug
        self.srm = AdaptiveSRM(1, out_channels=3, srm_mode=srm_mode, depthwise=False)

        def project(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, embed_channels, 1, bias=False),
                nn.GroupNorm(_group_norm_groups(embed_channels), embed_channels),
                nn.SiLU(inplace=True),
            )

        self.target_project = project(target_c)
        self.peer_project = project(peer_c)
        self.p4_project = project(p4_c)
        self.p5_project = project(p5_c)
        self.context_fuse = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_norm_groups(embed_channels), embed_channels),
            nn.SiLU(inplace=True),
        )
        self.prototype_assignment = nn.Conv2d(embed_channels, num_prototypes, 1)
        evidence_channels = 6  # K4/K5/K8, mean, max and kernel agreement
        self.verifier = nn.Sequential(
            nn.Linear(embed_channels * 2 + evidence_channels + 1, embed_channels),
            nn.SiLU(inplace=True),
            nn.Linear(embed_channels, 1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(target_c, target_c, 1, bias=False),
            nn.GroupNorm(_group_norm_groups(target_c), target_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(target_c, target_c, 3, padding=1, bias=False),
        )
        self.gamma = nn.Parameter(torch.full((1, target_c, 1, 1), float(gamma_init)))
        self.last_sparse_mask = None
        self.last_candidate_confidence = None

    @staticmethod
    def _sample(feature: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Sample BCHW feature at BxN normalized xy coordinates -> BxNxC."""
        sampled = F.grid_sample(
            feature, grid.unsqueeze(2), mode="bilinear", padding_mode="border", align_corners=True
        )
        return sampled.squeeze(-1).transpose(1, 2)

    def _candidates(self, response: torch.Tensor, size: tuple[int, int]):
        response = F.interpolate(response, size=size, mode="bilinear", align_corners=False)
        magnitude = response.mean(dim=1, keepdim=True)
        scale = magnitude.flatten(2).mean(-1, keepdim=True).unsqueeze(-1)
        scale = scale + magnitude.flatten(2).std(-1, keepdim=True).unsqueeze(-1)
        normalized = magnitude / scale.clamp_min(1e-6)
        peaks = normalized.eq(F.max_pool2d(normalized, 3, stride=1, padding=1))
        scores = normalized.masked_fill(~peaks, -1.0).flatten(1)
        k = min(self.num_candidates, scores.shape[1])
        top_scores, indices = scores.topk(k, dim=1)
        h, w = size
        y = torch.div(indices, w, rounding_mode="floor")
        x = indices.remainder(w)
        valid = (top_scores >= self.candidate_threshold).to(response.dtype)
        grid_x = x.to(response.dtype) * (2.0 / max(w - 1, 1)) - 1.0
        grid_y = y.to(response.dtype) * (2.0 / max(h - 1, 1)) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return response, grid, x, y, valid

    def _background_prototypes(self, p4: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
        p4_embed = self.p4_project(p4)
        p5_embed = F.interpolate(
            self.p5_project(p5), size=p4_embed.shape[-2:], mode="bilinear", align_corners=False
        )
        context = self.context_fuse(torch.cat((p4_embed, p5_embed), dim=1))
        assignment = self.prototype_assignment(context).flatten(2).softmax(dim=-1)
        return torch.einsum("bks,bcs->bkc", assignment, context.flatten(2))

    def _sparse_gaussians(
        self, x: torch.Tensor, y: torch.Tensor, confidence: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        h, w = size
        yy = torch.arange(h, device=x.device, dtype=confidence.dtype).view(1, 1, h, 1)
        xx = torch.arange(w, device=x.device, dtype=confidence.dtype).view(1, 1, 1, w)
        dx2 = (xx - x.to(confidence.dtype).unsqueeze(-1).unsqueeze(-1)).square()
        dy2 = (yy - y.to(confidence.dtype).unsqueeze(-1).unsqueeze(-1)).square()
        distance2 = dx2 + dy2
        sigma2 = self.gaussian_sigma ** 2
        local = distance2 <= (3.0 * self.gaussian_sigma) ** 2
        gaussian = torch.exp(-0.5 * distance2 / sigma2) * local
        return (gaussian * confidence.unsqueeze(-1).unsqueeze(-1)).amax(dim=1, keepdim=True)

    def forward(self, inputs):
        target, peer, p4, p5, image = inputs
        size = target.shape[-2:]
        response = self.srm(_grayscale(image)).abs()
        response, grid, x, y, valid = self._candidates(response, size)

        target_detail = self._sample(self.target_project(target), grid)
        peer_embed = F.interpolate(
            self.peer_project(peer), size=size, mode="bilinear", align_corners=False
        )
        peer_detail = self._sample(peer_embed, grid)
        residual = self._sample(response, grid)
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_max = residual.amax(dim=-1, keepdim=True)
        agreement = (residual > residual_mean).to(residual.dtype).mean(dim=-1, keepdim=True)
        evidence = torch.cat((residual, residual_mean, residual_max, agreement), dim=-1)

        prototypes = F.normalize(self._background_prototypes(p4, p5), dim=-1)
        detail_for_similarity = F.normalize((target_detail + peer_detail) * 0.5, dim=-1)
        similarity = torch.einsum("bnc,bkc->bnk", detail_for_similarity, prototypes).amax(dim=-1, keepdim=True)
        novelty = 0.5 * (1.0 - similarity)
        verifier_input = torch.cat((target_detail, peer_detail, evidence, novelty), dim=-1)
        confidence = torch.sigmoid(self.verifier(verifier_input)).squeeze(-1) * valid
        sparse_mask = self._sparse_gaussians(x, y, confidence, size)

        output = target + self.gamma * sparse_mask * self.refine(target)
        self.last_sparse_mask = sparse_mask.detach()
        self.last_candidate_confidence = confidence.detach()
        return (output, sparse_mask, confidence) if self.debug else output
