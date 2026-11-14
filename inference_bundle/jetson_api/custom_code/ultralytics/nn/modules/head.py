# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import TORCH_1_11, fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Proto26, RealNVP, Residual, SwiGLUFFN
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init
from .ftsc import AnchorFreeFTSCCalibrator

__all__ = (
    "OBB",
    "Classify",
    "Detect",
    "DetectClsAttention",
    "P3NUDFLDetect",
    "P2NUDFLDetect",
    "Pose",
    "RTDETRDecoder",
    "Segment",
    "SemanticSegment",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
    "v10GCTSDetect",
    "v10GCTSP3NUDFLDetect",
    "v10P3NUDFLDetect",
)

P2_NUDFL_BINS = (
    0.0, 0.35, 0.70, 1.05, 1.40, 1.80, 2.30, 2.90,
    3.60, 4.50, 5.60, 6.90, 8.40, 10.20, 12.40, 15.00,
)
P3_NUDFL_BINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0)


class BoxLocalDetail(nn.Module):
    """Lightweight local-detail adapter for box regression features."""

    def __init__(self, c: int, scale: float = 0.25, kernel: int = 3, gate: bool = True):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)
        self.edge = nn.Sequential(Conv(c, c, 3), Conv(c, c, 1))
        self.use_gate = gate
        self.gate = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid()) if gate else nn.Identity()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail = self.edge(x - self.pool(x))
        gate = self.gate(x) if self.use_gate else 1.0
        return x + self.scale * gate * detail


class P2OffsetRegression(nn.Module):
    """Four-side sub-cell sampling regression head for the P2 feature map."""

    def __init__(self, old_head: nn.Sequential, reg_max: int, rho: float = 0.5):
        super().__init__()
        self.reg_max = reg_max
        self.rho = rho
        self.stem = nn.Sequential(old_head[0], old_head[1])
        channels = old_head[1].conv.out_channels
        self.offset = nn.Sequential(DWConv(channels, channels, 3), nn.Conv2d(channels, 8, 1))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        old_logits = old_head[-1]
        self.sides = nn.ModuleList(nn.Conv2d(channels, reg_max, 1) for _ in range(4))
        with torch.no_grad():
            for side, start in zip(self.sides, range(0, 4 * reg_max, reg_max)):
                side.weight.copy_(old_logits.weight[start : start + reg_max])
                side.bias.copy_(old_logits.bias[start : start + reg_max])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.stem(x)
        b, _, h, w = feature.shape
        offset = self.rho * self.offset(feature).tanh().reshape(b, 4, 2, h, w)
        yy, xx = torch.meshgrid(
            torch.arange(h, device=feature.device, dtype=feature.dtype),
            torch.arange(w, device=feature.device, dtype=feature.dtype),
            indexing="ij",
        )
        base = torch.stack((2 * (xx + 0.5) / w - 1, 2 * (yy + 0.5) / h - 1), dim=-1)
        grids = base[None, None] + torch.stack(
            (2 * offset[:, :, 0] / w, 2 * offset[:, :, 1] / h), dim=-1
        )
        sampled = F.grid_sample(
            feature[:, None].expand(-1, 4, -1, -1, -1).reshape(4 * b, -1, h, w),
            grids.reshape(4 * b, h, w, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).reshape(b, 4, -1, h, w)
        return torch.cat([self.sides[i](sampled[:, i]) for i in range(4)], dim=1)


class Detect(nn.Module):
    """YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9/v11 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    max_det = 300  # max_det
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(
        self,
        nc: int = 80,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
        cls_geometry_fuse: bool = False,
        cls_geometry_mode: str = "add",
        cls_geometry_detach: bool = True,
        cls_deform_geometry: bool = False,
        quality_head: bool = False,
        quality_score_mode: str = "cls_mul_q",
        quality_box_features: bool = False,
        quality_box_detach: bool = True,
        dfl_residual: bool = False,
        dfl_residual_scale: float = 0.25,
        box_detail_head: bool = False,
        box_detail_levels: list[int] | tuple[int, ...] | None = None,
        box_detail_scale: float = 0.25,
        box_detail_kernel: int = 3,
        box_detail_gate: bool = True,
        p2_offset_regression: bool = False,
        p1_reg_injection: bool = False,
    ):
        """Initialize the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.p1_reg_injection = bool(p1_reg_injection)
        if self.p1_reg_injection:
            self.nl = len(ch) - 1
            ch_detect = ch[:-1]
            c_p2, c_p1 = ch[0], ch[-1]
            self.p1_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.p1_local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
            self.p1_proj = nn.Conv2d(c_p1, c_p2, kernel_size=1, bias=False)
            self.p1_zero_conv = nn.Conv2d(c_p2, c_p2, kernel_size=1, bias=False)
            nn.init.zeros_(self.p1_zero_conv.weight)
            self.p1_alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        else:
            self.nl = len(ch)
            ch_detect = ch

        self.reg_max = reg_max  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.cls_geometry_fuse = bool(cls_geometry_fuse)
        self.cls_geometry_mode = str(cls_geometry_mode).lower()
        if self.cls_geometry_mode not in {"add", "concat"}:
            raise ValueError("cls_geometry_mode must be 'add' or 'concat'.")
        self.cls_geometry_detach = bool(cls_geometry_detach)
        self.cls_deform_geometry = bool(cls_deform_geometry)
        if self.cls_deform_geometry:
            raise NotImplementedError("cls_deform_geometry is reserved for a future VFNet-like experiment.")
        self.quality_head = bool(quality_head)
        self.quality_score_mode = str(quality_score_mode)
        if self.quality_score_mode not in {"cls_mul_q", "sqrt_cls_mul_q", "cls_mul_q2"}:
            raise ValueError("quality_score_mode must be 'cls_mul_q', 'sqrt_cls_mul_q', or 'cls_mul_q2'.")
        self.quality_box_features = bool(quality_box_features)
        self.quality_box_detach = bool(quality_box_detach)
        self.dfl_residual = bool(dfl_residual)
        self.dfl_residual_scale = float(dfl_residual_scale)
        self.box_detail_head = bool(box_detail_head)
        self.box_detail_levels = set(range(self.nl) if box_detail_levels is None else box_detail_levels) if self.box_detail_head else set()
        c2, c3 = max((16, ch_detect[0] // 4, self.reg_max * 4)), max(ch_detect[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch_detect
        )
        self.box_detail = nn.ModuleList(
            BoxLocalDetail(x, box_detail_scale, box_detail_kernel, box_detail_gate)
            if i in self.box_detail_levels
            else nn.Identity()
            for i, x in enumerate(ch_detect)
        )
        self.cv2_residual = self.init_dfl_residual_heads(ch_detect) if self.dfl_residual else nn.ModuleList()
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch_detect)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch_detect
            )
        )
        if self.cls_geometry_fuse:
            cls_channels = [m[-1].in_channels for m in self.cv3]
            self.cls_geometry_embed = nn.ModuleList(nn.Conv2d(4, c, 1) for c in cls_channels)
            self.cls_geometry_fuse_conv = (
                nn.ModuleList(Conv(c * 2, c, 1) for c in cls_channels)
                if self.cls_geometry_mode == "concat"
                else nn.ModuleList(nn.Identity() for _ in cls_channels)
            )
        else:
            self.cls_geometry_embed = nn.ModuleList()
            self.cls_geometry_fuse_conv = nn.ModuleList()
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        # EXPERIMENTAL: localization quality map heads. These parameters live
        # on the model so the optimizer can update them during LQM training.
        self.loc_quality_enabled = False
        self.loc_cv = nn.ModuleList(nn.Conv2d(x, 1, 1) for x in ch)
        # EXPERIMENTAL: true IoU quality heads. Unlike loc_quality, this branch
        # is inference-visible and learns assigned predicted-box IoU.
        quality_ch = 14 if self.quality_box_features else 0
        self.cvq = nn.ModuleList(
            nn.Sequential(Conv(x + quality_ch, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 1, 1)) for x in ch
        )

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            if self.dfl_residual:
                self.one2one_cv2_residual = copy.deepcopy(self.cv2_residual)
            if self.cls_geometry_fuse:
                self.one2one_cls_geometry_embed = copy.deepcopy(self.cls_geometry_embed)
                self.one2one_cls_geometry_fuse_conv = copy.deepcopy(self.cls_geometry_fuse_conv)
        if self.nl == 4 and p2_offset_regression:
            self.cv2[0] = P2OffsetRegression(self.cv2[0], self.reg_max)
            if end2end:
                self.one2one_cv2[0] = P2OffsetRegression(self.one2one_cv2[0], self.reg_max)

    @staticmethod
    def init_dfl_residual_heads(ch: tuple) -> nn.ModuleList:
        """Create zero-initialized DFL residual heads."""
        heads = nn.ModuleList(nn.Conv2d(x, 4, 1) for x in ch)
        for head in heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        return heads

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        out = dict(box_head=self.cv2, cls_head=self.cv3)
        if getattr(self, "dfl_residual", False):
            out.update(box_residual_head=self.cv2_residual)
        if getattr(self, "quality_head", False):
            out.update(quality_head=self.cvq)
        if getattr(self, "cls_geometry_fuse", False):
            out.update(geom_embed=self.cls_geometry_embed, geom_fuse=self.cls_geometry_fuse_conv)
        return out

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        out = dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)
        if getattr(self, "dfl_residual", False):
            out.update(box_residual_head=self.one2one_cv2_residual)
        if getattr(self, "cls_geometry_fuse", False):
            out.update(geom_embed=self.one2one_cls_geometry_embed, geom_fuse=self.one2one_cls_geometry_fuse_conv)
        return out

    @property
    def end2end(self):
        """Checks if the model has one2one for v3/v5/v8/v9/v11 backward compatibility."""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self._end2end = value

    def _geometry_dist_map(self, box_logits: torch.Tensor) -> torch.Tensor:
        """Return DFL expected l/t/r/b distance maps from box logits."""
        bs, _, h, w = box_logits.shape
        dist = box_logits.view(bs, 4, self.reg_max, h, w).softmax(2)
        proj = torch.arange(self.reg_max, device=box_logits.device, dtype=box_logits.dtype).view(1, 1, -1, 1, 1)
        dist = (dist * proj).sum(2)
        dist = dist / max(self.reg_max - 1, 1)
        return dist.detach() if self.cls_geometry_detach else dist

    def _quality_box_summary(self, box_logits: torch.Tensor) -> torch.Tensor:
        """Return expected l/t/r/b, entropy, peakness, width, and height maps for the quality head."""
        bs, _, h, w = box_logits.shape
        prob = box_logits.view(bs, 4, self.reg_max, h, w).softmax(2)
        proj = torch.arange(self.reg_max, device=box_logits.device, dtype=box_logits.dtype).view(1, 1, -1, 1, 1)
        expected = (prob * proj).sum(2) / max(self.reg_max - 1, 1)
        entropy = -(prob * prob.clamp_min(1e-9).log()).sum(2) / max(math.log(self.reg_max), 1e-9)
        peak = prob.max(2).values
        width = (expected[:, 0:1] + expected[:, 2:3]) * 0.5
        height = (expected[:, 1:2] + expected[:, 3:4]) * 0.5
        summary = torch.cat((expected, entropy, peak, width, height), dim=1)
        return summary.detach() if self.quality_box_detach else summary

    def _quality_logits(
        self, quality_branch: torch.nn.Module, feature: torch.Tensor, box_logits: torch.Tensor | None
    ) -> torch.Tensor:
        """Return quality logits, optionally conditioned on DFL/box summary maps."""
        if getattr(self, "quality_box_features", False):
            if box_logits is None:
                raise ValueError("quality_box_features=True requires box logits for the quality head.")
            feature = torch.cat((feature, self._quality_box_summary(box_logits)), dim=1)
        return quality_branch(feature)

    def _geometry_cls_logits(
        self,
        cls_branch: torch.nn.Module,
        cls_input: torch.Tensor,
        dist_map: torch.Tensor,
        geom_embed: torch.nn.Module,
        geom_fuse: torch.nn.Module,
    ) -> torch.Tensor:
        """Return class logits after fusing an l/t/r/b geometry cue into cls features."""
        cls_feat = cls_input
        for layer in list(cls_branch.children())[:-1]:
            cls_feat = layer(cls_feat)
        geom_feat = geom_embed(dist_map)
        if self.cls_geometry_mode == "add":
            cls_feat = cls_feat + geom_feat
        else:
            cls_feat = geom_fuse(torch.cat((cls_feat, geom_feat), dim=1))
        return cls_branch[-1](cls_feat)

    def _forward_cls_branch(
        self, level: int, cls_branch: torch.nn.Module, cls_input: torch.Tensor
    ) -> torch.Tensor:
        """Run one classification branch, allowing subclasses to refine task-specific intermediate features."""
        return cls_branch(cls_input)

    def forward_head(
        self,
        x: list[torch.Tensor],
        cls_x: list[torch.Tensor] | None = None,
        box_head: torch.nn.Module = None,
        cls_head: torch.nn.Module = None,
        quality_head: torch.nn.Module = None,
        box_residual_head: torch.nn.Module = None,
        geom_embed: torch.nn.Module = None,
        geom_fuse: torch.nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        cls_x = x if cls_x is None else cls_x
        bs = x[0].shape[0]  # batch size
        box_features = [self.box_detail[i](x[i]) for i in range(self.nl)]
        if not getattr(self, "cls_geometry_fuse", False):
            boxes_per_level = [box_head[i](box_features[i]) for i in range(self.nl)]
            boxes = torch.cat([b.view(bs, 4 * self.reg_max, -1) for b in boxes_per_level], dim=-1)
            scores = torch.cat(
                [self._forward_cls_branch(i, cls_head[i], cls_x[i]).view(bs, self.nc, -1) for i in range(self.nl)],
                dim=-1,
            )
        else:
            boxes_per_level, scores_per_level = [], []
            for i in range(self.nl):
                box_logits = box_head[i](box_features[i])
                boxes_per_level.append(box_logits)
                dist_map = self._geometry_dist_map(box_logits)
                cls_logits = self._geometry_cls_logits(cls_head[i], cls_x[i], dist_map, geom_embed[i], geom_fuse[i])
                scores_per_level.append(cls_logits.view(bs, self.nc, -1))
            boxes = torch.cat([b.view(bs, 4 * self.reg_max, -1) for b in boxes_per_level], dim=-1)
            scores = torch.cat(scores_per_level, dim=-1)
        if getattr(self, "capture_dfl_diagnostics", False):
            self.last_p3_box_logits = boxes_per_level[0].detach()
        out = dict(boxes=boxes, scores=scores, feats=x)
        if getattr(self, "dfl_residual", False) and box_residual_head is not None:
            out["dfl_residual"] = torch.cat(
                [box_residual_head[i](box_features[i]).view(bs, 4, -1) for i in range(self.nl)], dim=-1
            )
        if getattr(self, "quality_head", False) and quality_head is not None:
            out["quality_logits"] = torch.cat(
                [self._quality_logits(quality_head[i], x[i], boxes_per_level[i]).view(bs, 1, -1) for i in range(self.nl)],
                dim=-1,
            )
        # EXPERIMENTAL: LQM maps are train-time auxiliary outputs only.
        if self.training and getattr(self, "loc_quality_enabled", False):
            out["loc_maps"] = [self.loc_cv[i](x[i]) for i in range(self.nl)]
        return out

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if getattr(self, "p1_reg_injection", False):
            x_p1 = x[-1]
            x_detect = x[:-1]

            # Local subtraction detail extraction on P1
            p1_down = self.p1_downsample(x_p1)
            local_avg = self.p1_local_pool(p1_down)
            L = torch.clamp(p1_down - local_avg, min=0.0)

            # Inject only into regression branch of level 0 (P2)
            p2_reg = x_detect[0] + self.p1_alpha * self.p1_zero_conv(self.p1_proj(L))
            box_x = [p2_reg] + [x_detect[i] for i in range(1, self.nl)]
            cls_x = x_detect

            preds = self.forward_head(box_x, cls_x=cls_x, **self.one2many)
            if self.end2end:
                box_x_detach = [xi.detach() for xi in box_x]
                cls_x_detach = [xi.detach() for xi in cls_x]
                one2one = self.forward_head(box_x_detach, cls_x=cls_x_detach, **self.one2one)
                preds = {"one2many": preds, "one2one": one2one}
        else:
            preds = self.forward_head(x, **self.one2many)
            if self.end2end:
                x_detach = [xi.detach() for xi in x]
                one2one = self.forward_head(x_detach, **self.one2one)
                preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(
        self, x: dict[str, torch.Tensor], return_quality_debug: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (dict[str, torch.Tensor]): Dictionary of predictions from detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        dbox = self._get_decode_boxes(x)
        scores = x["scores"].sigmoid()
        quality = None
        if getattr(self, "quality_head", False) and "quality_logits" in x:
            quality = x["quality_logits"].sigmoid()
            if self.quality_score_mode == "sqrt_cls_mul_q":
                scores = scores.clamp_min(0).sqrt() * quality
            elif self.quality_score_mode == "cls_mul_q2":
                scores = scores * quality.square()
            else:
                scores = scores * quality
        out = torch.cat((dbox, scores), 1)
        if return_quality_debug:
            debug = {"boxes": dbox, "final_scores": scores}
            if quality is not None:
                cls_scores = x["scores"].sigmoid()
                debug.update(cls_scores=cls_scores, q_scores=quality)
            return out, debug
        return out

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        bin_values = getattr(self, "p2_dfl_bins", getattr(self, "p3_dfl_bins", None))
        if bin_values is not None:
            boxes = x["boxes"]
            b, _, anchors = boxes.shape
            probability = boxes.view(b, 4, self.reg_max, anchors).softmax(2)
            uniform = torch.arange(self.reg_max, device=boxes.device, dtype=boxes.dtype).view(1, 1, -1, 1)
            custom = bin_values.to(device=boxes.device, dtype=boxes.dtype).view(1, 1, -1, 1)
            is_p2 = (self.strides.view(1, 1, 1, -1) == self.stride.min()).to(boxes.dtype)
            values = uniform + is_p2 * (custom - uniform)
            dist = (probability * values).sum(2)
        else:
            dist = self.dfl(x["boxes"])
        if getattr(self, "dfl_residual", False) and "dfl_residual" in x:
            residual = x["dfl_residual"].tanh() * self.dfl_residual_scale
            dist = (dist + residual).clamp(min=0, max=max(self.reg_max - 1, 0))
        dbox = self.decode_bboxes(dist, self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # from
            if isinstance(a, HVDecoupledRegression):
                for output in (a.horizontal[-1], a.vertical[-1]):
                    output.bias.data[:] = 2.0
            elif isinstance(a, P2OffsetRegression):
                for side in a.sides:
                    side.bias.data[:] = 2.0
            else:
                a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
        if getattr(self, "quality_head", False):
            for q in self.cvq:
                q[-1].bias.data[:] = math.log(0.01 / 0.99)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # from
                if isinstance(a, P2OffsetRegression):
                    for side in a.sides:
                        side.bias.data[:] = 2.0
                else:
                    a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x1, y1, x2, y2, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,80)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = max_det if self.export else min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cls_geometry_embed = self.cls_geometry_fuse_conv = None


class P3NUDFLDetect(Detect):
    """Detect head with the legacy non-uniform DFL codebook at its finest level."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("p3_dfl_bins", torch.tensor(P3_NUDFL_BINS), persistent=True)


class P2NUDFLDetect(Detect):
    """Detect head with the fixed non-uniform DFL codebook at P2."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("p2_dfl_bins", torch.tensor(P2_NUDFL_BINS), persistent=True)


class FTSCF5Calibrator(nn.Module):
    """Lightweight F5 supervision calibrator used after TAL assignment only.

    The module has no inference path.  It follows the YOLO-FTSC F5 recipe:
    position-Gaussian and detached DFL-distribution evidence are centered per
    GT, clipped in log space and applied as residual positive-loss weights.
    """

    def __init__(self, config: dict, reg_max: int) -> None:
        super().__init__()
        config = dict(config or {})
        if str(config.get("policy", "f5")).lower() != "f5":
            raise ValueError("This project supports FTSC policy='f5' only.")
        self.reg_max = int(reg_max)
        self.position_alpha = float(config.get("position_alpha", 6.0))
        self.entropy_tau = float(config.get("dfl_entropy_tau", 1.0))
        self.log_clip = float(config.get("log_clip", 0.35))
        self.warmup_epochs = int(config.get("warmup_epochs", 5))
        self.ramp_epochs = max(int(config.get("ramp_epochs", 10)), 1)
        self.apply_cls = bool(config.get("apply_cls", True))
        self.apply_box = bool(config.get("apply_box", True))
        self.apply_dfl = bool(config.get("apply_dfl", True))
        # F5 uses a learnable position strength and fixed DFL strength 1.0.
        self.position_strength_logit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("dfl_strength", torch.tensor(float(config.get("dfl_strength", 1.0))))
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _center_per_gt(values: torch.Tensor, fg_mask: torch.Tensor, target_gt_idx: torch.Tensor) -> torch.Tensor:
        groups = target_gt_idx[fg_mask].long()
        if not values.numel():
            return values
        n_groups = int(groups.max().item()) + 1
        sums = values.new_zeros(n_groups).scatter_add_(0, groups, values)
        counts = values.new_zeros(n_groups).scatter_add_(0, groups, torch.ones_like(values))
        return values - sums[groups] / counts[groups].clamp_min(1.0)

    def _ramp(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        return min(1.0, (epoch - self.warmup_epochs + 1) / self.ramp_epochs)

    def forward(self, anchor_points_px, target_bboxes_px, target_gt_idx, fg_mask, pred_distri, epoch: int = 0):
        if not fg_mask.any():
            empty = pred_distri.new_empty(0)
            return {"cls": empty, "box": empty, "dfl": empty, "regularization": pred_distri.sum() * 0.0}
        points = anchor_points_px.unsqueeze(0).expand(target_bboxes_px.shape[0], -1, -1)[fg_mask]
        boxes = target_bboxes_px[fg_mask]
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6)
        position = -0.5 * ((points - centers) / (sizes / self.position_alpha).clamp_min(1e-6)).square().sum(-1)
        logits = pred_distri[fg_mask].reshape(-1, 4, self.reg_max).float()
        probs = logits.softmax(-1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean(-1) / math.log(self.reg_max)
        dfl = (-self.entropy_tau * entropy).detach().to(position.dtype)
        position = self._center_per_gt(position, fg_mask, target_gt_idx)
        dfl = self._center_per_gt(dfl, fg_mask, target_gt_idx)
        strength = 2.0 * self.position_strength_logit.sigmoid()
        log_weight = (strength * position + self.dfl_strength * dfl).clamp(-self.log_clip, self.log_clip)
        weight = 1.0 + self._ramp(int(epoch)) * (log_weight.exp() - 1.0)
        self.last_metrics = {
            "ftsc_positive_count": float(weight.numel()), "ftsc_position_strength": float(strength.detach()),
            "ftsc_weight_mean": float(weight.detach().mean()), "ftsc_dfl_entropy_mean": float(entropy.detach().mean()),
        }
        one = torch.ones_like(weight)
        return {"cls": weight if self.apply_cls else one, "box": weight if self.apply_box else one,
                "dfl": weight if self.apply_dfl else one,
                "regularization": 1e-4 * (strength - 1.0).square()}


# Keep the historical local symbol for YAML/parser compatibility, but use the
# canonical FTSC implementation shared with the reference FTSC repository.
# This alias is intentionally placed after the legacy class definition so old
# imports remain valid while all new heads instantiate AnchorFreeFTSCCalibrator.
FTSCF5Calibrator = AnchorFreeFTSCCalibrator


class SRMClsDetect(Detect):
    """Detect with SRM-guided classification at P2 and ordinary deeper levels.

    The legacy two-input form ``[F2_reg, F2_cls]`` remains P2-only.  The
    multi-level form ``[F2_reg, F2_cls, P3, P4]`` keeps the ASRM/semantic gate
    exclusively on the P2 classification path while P3 and P4 use their
    unmodified feature for both regression and classification.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = (), *args, **kwargs):
        if len(ch) not in {2, 4}:
            raise ValueError(f"SRMClsDetect expects [F2_reg, F2_cls] or [F2_reg, F2_cls, P3, P4], got {ch}")
        if ch[0] != ch[1]:
            raise ValueError(f"SRMClsDetect requires matching F2 channels, got {ch}")
        # ``parse_model`` always appends the YAML-level FTSC field for this
        # head. Pop it even when it is ``None`` so legacy SRMClsDetect YAMLs
        # retain the exact Detect constructor arity.
        ftsc = args[-1] if args else None
        if args:
            args = args[:-1]
        super().__init__(nc, reg_max, end2end, (ch[0], *ch[2:]), *args, **kwargs)
        self.ftsc_calibrator = FTSCF5Calibrator(ftsc, reg_max) if ftsc and ftsc.get("enabled", True) else None
        self.last_reg_feature = None
        self.last_cls_feature = None

    def forward(self, x: list[torch.Tensor]):
        if len(x) not in {2, 4}:
            raise ValueError("SRMClsDetect forward expects [F2_reg, F2_cls] or [F2_reg, F2_cls, P3, P4]")
        reg_x, cls_x, *deep_x = x
        reg_features = [reg_x, *deep_x]
        cls_features = [cls_x, *deep_x]
        self.last_reg_feature = reg_x.detach()
        self.last_cls_feature = cls_x.detach()
        preds = self.forward_head(reg_features, cls_x=cls_features, **self.one2many)
        if self.end2end:
            one2one = self.forward_head(
                [feature.detach() for feature in reg_features],
                cls_x=[feature.detach() for feature in cls_features],
                **self.one2one,
            )
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)


class FTSCDetect(Detect):
    """Ordinary multi-level Detect with training-only FTSC-F5 calibration."""

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = (), *args, **kwargs):
        ftsc = args[-1] if args else None
        if args:
            args = args[:-1]
        super().__init__(nc, reg_max, end2end, ch, *args, **kwargs)
        self.ftsc_calibrator = FTSCF5Calibrator(ftsc, reg_max) if ftsc and ftsc.get("enabled", True) else None


class ContrastClsFTSCDetect(FTSCDetect):
    """FTSC-compatible three-level head with separate cls features.

    YAML input order is ``[P2_reg, P2_cls, P3_reg, P3_cls, P4_reg, P4_cls]``.
    Only the classification branch sees contrast guidance; regression tensors
    are passed to the inherited box heads unchanged.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = (), *args, **kwargs):
        if len(ch) != 6:
            raise ValueError(f"ContrastClsFTSCDetect expects six tensors, got {ch}")
        if any(ch[i] != ch[i + 1] for i in (0, 2, 4)):
            raise ValueError(f"Contrast cls/reg channel pairs must match, got {ch}")
        super().__init__(nc, reg_max, end2end, (ch[0], ch[2], ch[4]), *args, **kwargs)
        self.last_reg_features = None
        self.last_cls_features = None

    def forward(self, x: list[torch.Tensor]):
        if len(x) != 6:
            raise ValueError("ContrastClsFTSCDetect forward expects [P2r,P2c,P3r,P3c,P4r,P4c]")
        reg_features = [x[0], x[2], x[4]]
        cls_features = [x[1], x[3], x[5]]
        self.last_reg_features = tuple(feature.detach() for feature in reg_features)
        self.last_cls_features = tuple(feature.detach() for feature in cls_features)
        preds = self.forward_head(reg_features, cls_x=cls_features, **self.one2many)
        if self.end2end:
            one2one = self.forward_head(
                [feature.detach() for feature in reg_features],
                cls_x=[feature.detach() for feature in cls_features],
                **self.one2one,
            )
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)


class HVDecoupledRegression(nn.Module):
    """Shared P2 regression stem with horizontal and vertical DFL towers."""

    def __init__(self, channels: int, reg_max: int) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.shared = Conv(channels, 64, 3)
        self.horizontal = nn.Sequential(Conv(64, 32, 3), nn.Conv2d(32, 2 * reg_max, 1))
        self.vertical = nn.Sequential(Conv(64, 32, 3), nn.Conv2d(32, 2 * reg_max, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        horizontal = self.horizontal(shared).unflatten(1, (2, self.reg_max))
        vertical = self.vertical(shared).unflatten(1, (2, self.reg_max))
        return torch.stack((horizontal[:, 0], vertical[:, 0], horizontal[:, 1], vertical[:, 1]), dim=1).flatten(1, 2)


class HVDecoupledDetect(Detect):
    """P2-only Detect head with direction-specific regression representations."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.nl != 1:
            raise ValueError(f"HVDecoupledDetect requires one P2 detection level, got {self.nl}")
        if self.end2end:
            raise ValueError("HVDecoupledDetect does not support end2end mode")
        channels = self.cv2[0][0].conv.in_channels
        self.cv2[0] = HVDecoupledRegression(channels, self.reg_max)


class Segment(Detect):
    """YOLO Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, mask_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x1, y1, x2, y2, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class Segment26(Segment):
    """YOLO26 Segment head for segmentation models.

    This class extends the Segment head with Proto26 for mask prediction in instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto26): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment26(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self) -> None:
        """Remove the one2many head and extra part of proto module for inference optimization."""
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models.

    This class extends the Detect head to include oriented bounding box prediction with rotation angles.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        decode_bboxes: Decode rotated bounding boxes.

    Examples:
        Create an OBB detection head
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`.

        Args:
            nc (int): Number of classes.
            ne (int): Number of extra parameters.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, angle_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, angle_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with rotation angles."""
        # For decode_bboxes convenience
        self.angle = x["angle"]
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and angles."""
        preds = super().forward_head(x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits
            angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
            preds["angle"] = angle
        return preds

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + ne) with last dimension
                format [x, y, w, h, class_probs, angle].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 7) and last
                dimension format [x, y, w, h, max_class_prob, class_index, angle].
        """
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class OBB26(OBB):
    """YOLO26 OBB detection head for detection with rotation models. This class extends the OBB head with modified angle
    processing that outputs raw angle predictions without sigmoid transformation, compared to the original
    OBB class.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward_head: Concatenate and return predicted bounding boxes, class probabilities, and raw angles.

    Examples:
        Create an OBB26 detection head
        >>> obb26 = OBB26(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb26(x)
    """

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and raw angles."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits (raw output without sigmoid transformation)
            preds["angle"] = angle
        return preds


class Pose(Detect):
    """YOLO Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, pose_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with keypoints."""
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, pose_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = super().forward_head(x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nk) with last dimension
                format [x1, y1, x2, y2, class_probs, keypoints].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + self.nk) and
                last dimension format [x1, y1, x2, y2, max_class_prob, class_index, keypoints].
        """
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Pose26(Pose):
    """YOLO26 Pose head for keypoints models.

    This class extends the Pose head with normalizing flow for keypoint prediction in pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose26(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch)

        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        self.nk_sigma = kpt_shape[0] * 2  # sigma_x, sigma_y for each keypoint
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            pose_head=self.cv4,
            kpts_head=self.cv4_kpts,
            kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            pose_head=self.one2one_cv4,
            kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat([kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
            if self.training:
                preds["kpts_sigma"] = torch.cat(
                    [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                )
        return preds

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        super().fuse()
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            # NCNN fix
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2).

    This class implements a classification head that transforms feature maps into class predictions.

    Attributes:
        export (bool): Export mode flag.
        conv (Conv): Convolutional layer for feature transformation.
        pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        drop (nn.Dropout): Dropout layer for regularization.
        linear (nn.Linear): Linear layer for final classification.

    Methods:
        forward: Perform forward pass on input feature maps.

    Examples:
        Create a classification head
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # export mode

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output classes.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """Perform forward pass on input feature maps."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to incorporate text embeddings for enhanced semantic understanding in
    object detection tasks.

    Attributes:
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        bias_init: Initialize detection head biases.

    Examples:
        Create a WorldDetect head
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        embed: int = 512,
        with_bn: bool = False,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> dict[str, torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        feats = [xi.clone() for xi in x]  # save original features for anchor generation
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        bs = x[0].shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
        boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)
        preds = dict(boxes=boxes, scores=scores, feats=feats)
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class LRPCHead(nn.Module):
    """Lightweight Region Proposal and Classification Head for efficient object detection.

    This head combines region proposal filtering with classification to enable efficient detection with dynamic
    vocabulary support.

    Attributes:
        vocab (nn.Module): Vocabulary/classification layer.
        pf (nn.Module): Proposal filter module.
        loc (nn.Module): Localization module.
        enabled (bool): Whether the head is enabled.

    Methods:
        conv2linear: Convert a 1x1 convolutional layer to a linear layer.
        forward: Process classification and localization features to generate detection proposals.

    Examples:
        Create an LRPC head
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """Initialize LRPCHead with vocabulary, proposal filter, and localization components.

        Args:
            vocab (nn.Module): Vocabulary/classification module.
            pf (nn.Module): Proposal filter module.
            loc (nn.Module): Localization module.
            enabled (bool): Whether to enable the head functionality.
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return self.loc(loc_feat), cls_feat.transpose(-1, -2), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (
                loc_feat,
                cls_feat.flatten(2),
                torch.ones(cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool),
            )


class YOLOEDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding
    through text embeddings and visual prompt embeddings.

    Attributes:
        is_fused (bool): Whether the model is fused for inference.
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.
        reprta (Residual): Residual block for text prompt embeddings.
        savpe (SAVPE): Spatial-aware visual prompt embeddings module.
        embed (int): Embedding dimension.

    Methods:
        fuse: Fuse text features with model weights for efficient inference.
        get_tpe: Get text prompt embeddings with normalization.
        get_vpe: Get visual prompt embeddings with spatial awareness.
        forward_lrpc: Process features with fused text embeddings for prompt-free model.
        forward: Process features with class prompt embeddings to generate detections.
        bias_init: Initialize biases for detection heads.

    Examples:
        Create a YOLOEDetect head
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect([*x, cls_pe])
    """

    is_fused = False

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = ()
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        if end2end:
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # overwrite with new cv3
            self.one2one_cv4 = copy.deepcopy(self.cv4)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        if txt_feats is None:  # means eliminate one2many branch
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        if self.cv3 and self.cv4:
            self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """Fuse text prompt embeddings with model weights for efficient inference."""
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv = cls_h[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = conv

            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """Get text prompt embeddings with normalization."""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x[:3])
        return super().forward(x)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=x, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _get_decode_boxes(self, x):
        """Decode predicted bounding boxes for inference."""
        dbox = super()._get_decode_boxes(x)
        if hasattr(self, "lrpc"):
            dbox = dbox if self.export and not self.dynamic else dbox[..., x["index"]]
        return dbox

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, contrastive_head=self.one2one_cv4)

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """Concatenates and returns predicted bounding boxes, class probabilities, and contrastive scores."""
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [contrastive_head[i](cls_head[i](x[i]), x[-1]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        return dict(boxes=boxes, scores=scores, feats=x[:3])

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class YOLOESegment(YOLOEDetect):
    """YOLO segmentation head with text embedding capabilities.

    This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks with
    text-guided semantic understanding.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv5 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a YOLOESegment head
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment([*x, text])
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment with class count, mask parameters, and embedding dimensions.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cv5 = self.cv5 if not self.end2end else self.one2one_cv5
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        index = torch.cat(index)
        preds = dict(
            boxes=torch.cat(boxes, 2),
            scores=torch.cat(scores, 2),
            feats=x,
            index=index,
            mask_coefficient=mc * index.int() if self.export and not self.dynamic else mc[..., index],
        )
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        mask_head: torch.nn.Module,
        contrastive_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x1, y1, x2, y2, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        super().fuse(txt_feats)
        if txt_feats is None:  # means eliminate one2many branch
            self.cv5 = None
            if hasattr(self.proto, "fuse"):
                self.proto.fuse()
            return


class YOLOESegment26(YOLOESegment):
    """YOLOE-style segmentation head module using Proto26 for mask generation.

    This class extends the YOLOESegment functionality to include segmentation capabilities by integrating a Proto26
    generation module and convolutional layers to predict mask coefficients.

    Args:
        nc (int): Number of classes. Defaults to 80.
        nm (int): Number of masks. Defaults to 32.
        npr (int): Number of prototype channels. Defaults to 256.
        embed (int): Embedding dimensionality. Defaults to 512.
        with_bn (bool): Whether to use Batch Normalization. Defaults to False.
        reg_max (int): Maximum number of DFL channels. Defaults to 16.
        end2end (bool): Whether to use end-to-end detection mode. Defaults to False.
        ch (tuple[int, ...]): Input channels for each scale.

    Attributes:
        nm (int): Number of segmentation masks.
        npr (int): Number of prototype channels.
        proto (Proto26): Prototype generation module for segmentation.
        cv5 (nn.ModuleList): Convolutional layers for generating mask coefficients from features.
        one2one_cv5 (nn.ModuleList, optional): Deep copy of cv5 for end-to-end detection branches.
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment26 with class count, mask parameters, and embedding dimensions."""
        YOLOEDetect.__init__(self, nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = YOLOEDetect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto([xi.detach() for xi in x], return_semantic=False)  # mask protos

        if isinstance(preds, dict):  # training and validating during training
            if self.end2end and not hasattr(self, "lrpc"):  # not prompt-free
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class RTDETRDecoder(nn.Module):
    """Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.

    Attributes:
        export (bool): Export mode flag.
        hidden_dim (int): Dimension of hidden layers.
        nhead (int): Number of heads in multi-head attention.
        nl (int): Number of feature levels.
        nc (int): Number of classes.
        num_queries (int): Number of query points.
        num_decoder_layers (int): Number of decoder layers.
        input_proj (nn.ModuleList): Input projection layers for backbone features.
        decoder (DeformableTransformerDecoder): Transformer decoder module.
        denoising_class_embed (nn.Embedding): Class embeddings for denoising.
        num_denoising (int): Number of denoising queries.
        label_noise_ratio (float): Label noise ratio for training.
        box_noise_scale (float): Box noise scale for training.
        learnt_init_query (bool): Whether to learn initial query embeddings.
        tgt_embed (nn.Embedding): Target embeddings for queries.
        query_pos_head (MLP): Query position head.
        enc_output (nn.Sequential): Encoder output layers.
        enc_score_head (nn.Linear): Encoder score prediction head.
        enc_bbox_head (MLP): Encoder bbox prediction head.
        dec_score_head (nn.ModuleList): Decoder score prediction heads.
        dec_bbox_head (nn.ModuleList): Decoder bbox prediction heads.

    Methods:
        forward: Run forward pass and return bounding box and classification scores.

    Examples:
        Create an RTDETRDecoder
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # export mode
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries
        ndp: int = 4,  # num decoder points
        nh: int = 8,  # num head
        ndl: int = 6,  # num decoder layers
        d_ffn: int = 1024,  # dim of feedforward
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        # Training args
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """Initialize the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Dimension of hidden layers.
            nq (int): Number of query points.
            ndp (int): Number of decoder points.
            nh (int): Number of heads in multi-head attention.
            ndl (int): Number of decoder layers.
            d_ffn (int): Dimension of the feed-forward networks.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """Run the forward pass of the module, returning bounding box and classification scores for the input.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            outputs (tuple | torch.Tensor): During training, returns a tuple of bounding boxes, scores, and other
                metadata. During inference, returns a tensor of shape (bs, num_queries, 6) containing bounding boxes,
                confidence scores, and class labels.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        if self.training and dn_meta is None:
            # Touch denoising_class_embed so DDP sees it as used when batch has zero GTs.
            dec_bboxes = dec_bboxes + 0 * self.denoising_class_embed.weight.sum()
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, num_queries, 4), (bs, num_queries, nc)
        y = self.postprocess(dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid())
        return y if self.export else (y, x)

    def postprocess(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Post-process predictions to select top-k detections.

        Args:
            boxes (torch.Tensor): Predicted bounding boxes with shape (batch_size, num_queries, 4) in xywh format.
            scores (torch.Tensor): Class scores with shape (batch_size, num_queries, nc).

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, num_queries, 6) and last dimension format [cx,
                cy, w, h, max_class_prob, class_index].
        """
        scores, index = scores.flatten(1).topk(self.num_queries)
        # CoreML MIL lacks integer floor-div and mod lowering: use torch.div(rounding_mode="floor") and (index - q*nc).
        query_idx = torch.div(index, self.nc, rounding_mode="floor")
        boxes = boxes.gather(dim=1, index=query_idx.unsqueeze(-1).expand(-1, -1, 4).long())
        return torch.cat([boxes, scores[..., None], (index - query_idx * self.nc)[..., None].float()], dim=-1)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor bounding boxes for given shapes with specific grid size and validate them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells.
            dtype (torch.dtype, optional): Data type for tensors.
            device (str, optional): Device to create tensors on.
            eps (float, optional): Small value for numerical stability.

        Returns:
            anchors (torch.Tensor): Generated anchor boxes.
            valid_mask (torch.Tensor): Valid mask for anchors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """Process and return encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            feats (torch.Tensor): Processed features.
            shapes (list): List of feature map shapes.
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate and prepare the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes.

        Returns:
            embeddings (torch.Tensor): Query embeddings for decoder.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            enc_bboxes (torch.Tensor): Encoded bounding boxes.
            enc_scores (torch.Tensor): Encoded scores.
        """
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        # Prepare input for decoder
        features = self.enc_output(self.valid_mask * feats)  # bs, h*w, 256
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs*num_queries,)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs*num_queries,)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """Initialize or reset the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """v10 Detection head from https://arxiv.org/pdf/2405.14458.

    This class implements the YOLOv10 detection head with dual-assignment training and consistent dual predictions for
    improved efficiency and performance.

    Attributes:
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum number of detections.
        cv3 (nn.ModuleList): Light classification head layers.
        one2one_cv3 (nn.ModuleList): One-to-one classification head layers.

    Methods:
        __init__: Initialize the v10Detect object with specified number of classes and input channels.
        forward: Perform forward pass of the v10Detect module.
        bias_init: Initialize biases of the Detect module.
        fuse: Remove the one2many head for inference optimization.

    Examples:
        Create a v10Detect head
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Initialize the v10Detect object with the specified number of classes and input channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, end2end=True, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None


class v10P3NUDFLDetect(v10Detect):
    """YOLOv10 head using a fixed non-uniform DFL codebook only at P3."""

    def __init__(self, nc: int = 80, ch: tuple = ()) -> None:
        super().__init__(nc, ch)
        self.register_buffer("p3_dfl_bins", torch.tensor(P3_NUDFL_BINS), persistent=True)


class v10GCTSDetect(v10Detect):
    """YOLOv10 head with separate P2-guided classification and regression features."""

    def __init__(
        self,
        nc: int = 80,
        epsilon: float = 0.05,
        tiny_gate: bool = True,
        ch: tuple = (),
        detail_ratio: float = 0.25,
        select_weight: float = 0.1,
        coord_weight: float = 1.0,
        gate_weight: float = 0.1,
    ) -> None:
        if len(ch) != 4 or epsilon <= 0 or not 0 < detail_ratio <= 1:
            raise ValueError("v10GCTSDetect requires [P2, P3, P4, P5], positive epsilon, and valid detail_ratio")
        p2_channels, *detect_channels = ch
        super().__init__(nc, tuple(detect_channels))
        detail_channels = max(round(detect_channels[0] * detail_ratio), 1)
        self.p2_projection = nn.Sequential(nn.Conv2d(p2_channels, detail_channels, 1, bias=False), nn.SiLU())
        self.selector = nn.Conv2d(4 * detail_channels, 4, 1)
        self.cls_projection = nn.Conv2d(4 * detail_channels, detect_channels[0], 1)
        self.pos_projection = nn.Conv2d(7, detect_channels[0], 1)
        self.tiny_gate_head = nn.Conv2d(detect_channels[0], 1, 1)
        nn.init.zeros_(self.cls_projection.weight)
        nn.init.zeros_(self.cls_projection.bias)
        nn.init.zeros_(self.pos_projection.weight)
        nn.init.zeros_(self.pos_projection.bias)
        nn.init.zeros_(self.tiny_gate_head.weight)
        nn.init.zeros_(self.tiny_gate_head.bias)
        self.epsilon = float(epsilon)
        self.tiny_gate = bool(tiny_gate)
        self.select_weight = float(select_weight)
        self.coord_weight = float(coord_weight)
        self.gate_weight = float(gate_weight)
        self.capture_diagnostics = False
        self.last_gcts: dict[str, torch.Tensor] | None = None

    def __getstate__(self) -> dict:
        """Exclude transient autograd tensors when trainers deepcopy the model."""
        state = super().__getstate__().copy()
        state["last_gcts"] = None
        return state

    def _route(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        p2, p3, p4, p5 = x
        projected = self.p2_projection(p2)
        packed = F.pixel_unshuffle(projected, 2)
        if packed.shape[-2:] != p3.shape[-2:]:
            raise ValueError(f"GCTS P2/P3 spatial mismatch: {packed.shape[-2:]} vs {p3.shape[-2:]}")
        b, _, h, w = packed.shape
        candidates = packed.reshape(b, -1, 4, h, w)
        alpha = self.selector(packed).softmax(1)
        content = (candidates * alpha[:, None]).reshape(b, -1, h, w)
        cls_p3 = p3 + self.cls_projection(content)
        x_hat = alpha[:, 1:2] + alpha[:, 3:4]
        y_hat = alpha[:, 2:3] + alpha[:, 3:4]
        entropy = -(alpha * alpha.clamp_min(1e-9).log()).sum(1, keepdim=True) / math.log(4)
        position = torch.cat((alpha, x_hat - 0.5, y_hat - 0.5, entropy), 1)
        correction = self.epsilon * self.pos_projection(position).tanh()
        gate_logits = self.tiny_gate_head(p3) if self.tiny_gate else None
        gate = gate_logits.sigmoid() if gate_logits is not None else torch.ones_like(x_hat)
        box_p3 = p3 + gate * correction
        self.last_gcts = (
            {"alpha": alpha, "gate": gate, "gate_logits": gate_logits, "x_hat": x_hat, "y_hat": y_hat}
            if self.training or self.capture_diagnostics
            else None
        )
        return [box_p3, p4, p5], [cls_p3, p4, p5]

    def forward(self, x: list[torch.Tensor]):
        """Route P2 detail separately into YOLOv10 box and classification branches."""
        box_x, cls_x = self._route(x)
        preds = self.forward_head(box_x, cls_x=cls_x, **self.one2many)
        if self.end2end:
            one2one = self.forward_head(
                [xi.detach() for xi in box_x], cls_x=[xi.detach() for xi in cls_x], **self.one2one
            )
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _targets(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = self.last_gcts["alpha"]
        centers = batch["bboxes"][:, :2]
        coordinates = centers * centers.new_tensor((alpha.shape[-1], alpha.shape[-2]))
        coordinates[:, 0].clamp_(0, alpha.shape[-1] - 1e-6)
        coordinates[:, 1].clamp_(0, alpha.shape[-2] - 1e-6)
        cells = coordinates.floor().long()
        fractions = coordinates - cells
        x, y = fractions.unbind(1)
        q = torch.stack(((1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y), 1)
        return batch["batch_idx"].view(-1).long(), cells[:, 1], cells[:, 0], fractions, q

    def _gate_targets(
        self, batch: dict, batch_indices: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.last_gcts["gate"]
        state = torch.full(gate.shape[:1] + gate.shape[-2:], -2, device=gate.device, dtype=torch.int8)
        image_h, image_w = batch["img"].shape[-2:]
        sizes = torch.sqrt((batch["bboxes"][:, 2] * image_w).square() + (batch["bboxes"][:, 3] * image_h).square())
        for bi, y, x, size in zip(batch_indices.tolist(), ys.tolist(), xs.tolist(), sizes.tolist()):
            current = int(state[bi, y, x])
            if size < 20:
                state[bi, y, x] = 1
            elif size <= 24 and current != 1:
                state[bi, y, x] = -1
            elif current == -2:
                state[bi, y, x] = 0
        labeled = (state >= 0).nonzero(as_tuple=False)
        background = (state == -2).nonzero(as_tuple=False)
        if len(background) and len(labeled):
            pick = torch.linspace(0, len(background) - 1, min(len(background), len(labeled)), device=gate.device).long()
            labeled = torch.cat((labeled, background[pick]))
        targets = (state[labeled[:, 0], labeled[:, 1], labeled[:, 2]] == 1).to(gate.dtype)
        return labeled, targets

    def _gate_loss(self, batch: dict, batch_indices: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        gate = self.last_gcts["gate"]
        if not self.tiny_gate or not len(batch_indices):
            return gate.sum() * 0
        labeled, targets = self._gate_targets(batch, batch_indices, ys, xs)
        if not len(labeled):
            return gate.sum() * 0
        logits = self.last_gcts["gate_logits"][labeled[:, 0], 0, labeled[:, 1], labeled[:, 2]]
        return F.binary_cross_entropy_with_logits(logits, targets)

    def auxiliary_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Supervise selector geometry and the optional tiny-object gate."""
        if self.last_gcts is None or batch["bboxes"].numel() == 0:
            zero = self.selector.weight.sum() * 0
            self.last_gcts = None
            return zero, {"loss_gcts_v2_pos": zero.detach(), "loss_gcts_v2_gate": zero.detach()}
        bi, ys, xs, fractions, q = self._targets(batch)
        alpha = self.last_gcts["alpha"][bi, :, ys, xs]
        kl = (q * (q.clamp_min(1e-9).log() - alpha.clamp_min(1e-9).log())).sum(1).mean()
        expected = torch.stack((alpha[:, 1] + alpha[:, 3], alpha[:, 2] + alpha[:, 3]), 1)
        coordinate = F.smooth_l1_loss(expected, fractions)
        position_loss = self.select_weight * (kl + self.coord_weight * coordinate)
        gate_loss = self._gate_loss(batch, bi, ys, xs) * self.gate_weight
        loss = position_loss + gate_loss
        metrics = {
            "loss_gcts_v2_pos": position_loss.detach(),
            "loss_gcts_v2_gate": gate_loss.detach(),
            "gcts_v2_coord_mae": (expected - fractions).abs().mean().detach(),
            "gcts_v2_x_bias": (expected[:, 0] - fractions[:, 0]).mean().detach(),
        }
        self.last_gcts = None
        return loss, metrics


class v10GCTSP3NUDFLDetect(v10GCTSDetect):
    """GCTS v2 e05 head using the fixed non-uniform DFL codebook at P3."""

    def __init__(self, nc: int = 80, ch: tuple = ()) -> None:
        super().__init__(nc=nc, epsilon=0.05, tiny_gate=True, ch=ch)
        self.register_buffer("p3_dfl_bins", torch.tensor(P3_NUDFL_BINS), persistent=True)


class SemanticSegment(nn.Module):
    """YOLO semantic segmentation head for per-pixel classification.

    This head produces dense per-pixel class predictions. Unlike instance segmentation, no bounding boxes or instance
    masks are produced.

    Attributes:
        nc (int): Number of semantic classes.
        nl (int): Number of input feature levels.
        stride (torch.Tensor): Feature map strides.
        export (bool): Export mode flag.
        format (str): Export format.
        classifier (nn.Sequential): Final convolutional classifier head.
        aux_head (nn.Sequential | None): Auxiliary classifier on P4 for deep supervision.
    """

    export = False  # export mode
    format = None  # export format

    def __init__(self, nc=19, ch=()):
        """Initialize the semantic segmentation head.

        Args:
            nc (int): Number of semantic classes.
            ch (tuple): Tuple of channel sizes from neck feature maps (P3, P4).
        """
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.stride = torch.zeros(self.nl)

        c_mid = ch[0]  # use P3 channel width as intermediate dimension
        # Final classifier
        self.classifier = nn.Sequential(Conv(c_mid, c_mid, 3), nn.Conv2d(c_mid, nc, 1))
        # Auxiliary head on P4 (index 1) for training
        self.aux_head = nn.Sequential(Conv(ch[1], c_mid, 3), nn.Conv2d(c_mid, nc, 1)) if len(ch) > 1 else None

    def forward(self, x):
        """Forward pass: fuse multi-scale features and predict per-pixel classes.

        Args:
            x (list[torch.Tensor]): List of feature maps [P3, P4].

        Returns:
            (torch.Tensor | tuple): Logits of shape [B, nc, H/8, W/8] during training (or a (main, aux) tuple when
                aux_head is present) and inference. Export returns upsampled logits of shape [B, nc, H, W].
        """
        # Classify
        logits = self.classifier(x[0])  # [B, nc, H/8, W/8]
        if self.training:
            if self.aux_head is not None:
                return logits, self.aux_head(x[1])  # main + aux (P4)
            return logits
        if self.export:
            return F.interpolate(logits, scale_factor=8, mode="bilinear", align_corners=False)
        return logits


class P2ClsContext(nn.Module):
    """Zero-initialized multi-kernel context residual for a task-specific P2 classification feature."""

    def __init__(self, channels: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or channels // 2
        self.reduce = Conv(channels, hidden, 1)
        self.dw3 = DWConv(hidden, hidden, 3)
        self.dw5 = DWConv(hidden, hidden, 5)
        self.fuse = Conv(hidden * 2, channels, 1)
        from .block import CBAM

        self.attn = CBAM(channels, kernel_size=3)
        self.zero = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.zero.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        context = self.fuse(torch.cat((self.dw3(reduced), self.dw5(reduced)), dim=1))
        return x + self.zero(self.attn(context))


class P2RegLocal(nn.Module):
    """Zero-initialized local residual refiner for the P2 regression input."""

    def __init__(self, channels: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or channels // 2
        self.reduce = Conv(channels, hidden, 1)
        self.dw1 = DWConv(hidden, hidden, 3)
        self.pw = Conv(hidden, hidden, 1)
        self.dw2 = DWConv(hidden, hidden, 3)
        self.zero = nn.Conv2d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.zero.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.zero(self.dw2(self.pw(self.dw1(self.reduce(x)))))


class DetectClsAttention(Detect):
    """Detect head with class-specific attention at P2 (level 0)."""

    def __init__(self, *args, **kwargs) -> None:
        attn_type = kwargs.pop("attn_type", None)
        if attn_type is None and len(args) > 21:
            attn_type = args[-1]
            args = args[:-1]

        super().__init__(*args, **kwargs)
        self.attn_type = str(attn_type or "cbam").lower()
        c_p2 = self.cv2[0][0].conv.in_channels
        self.cls_mid = nn.Identity()
        if self.attn_type == "cbam":
            from .block import CBAM
            self.attn = CBAM(c_p2)
        elif self.attn_type == "kvca":
            from .block import KVCompressedTransformerEncoder
            self.attn = KVCompressedTransformerEncoder(c_p2, c_p2, num_heads=4, sr_ratio=8, mode="dwconv")
        elif self.attn_type == "kvca_block":
            from .block import KVCompressedAttention
            self.attn = KVCompressedAttention(c_p2, c_p2, num_heads=4, sr_ratio=8, mode="group_weight")
        elif self.attn_type == "context_mid_cbam":
            self.attn = nn.Identity()
            self.cls_mid = P2ClsContext(self.cv3[0][-1].in_channels)
        elif self.attn_type == "reg_local":
            self.attn = nn.Identity()
            self.box_detail[0] = P2RegLocal(c_p2)
        else:
            raise ValueError(f"Unsupported attn_type: {self.attn_type}")

    def _forward_cls_branch(self, level, cls_branch, cls_input):
        if self.attn_type != "context_mid_cbam" or level != 0:
            return cls_branch(cls_input)
        layers = list(cls_branch.children())
        if len(layers) != 3:
            raise RuntimeError(f"Expected a three-stage YOLOv8 classification branch, got {len(layers)} stages")
        return layers[2](layers[1](self.cls_mid(layers[0](cls_input))))

    def forward(self, x):
        """Forward pass applying attention only to classification branch of level 0 (P2)."""
        cls_x = [self.attn(x[0])] + [x[i] for i in range(1, self.nl)]

        preds = self.forward_head(x, cls_x=cls_x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            cls_x_detach = [self.attn(x_detach[0])] + [x_detach[i] for i in range(1, self.nl)]
            one2one = self.forward_head(x_detach, cls_x=cls_x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}

        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)
