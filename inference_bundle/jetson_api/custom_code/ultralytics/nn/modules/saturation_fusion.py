"""Lightweight saturation-guided residual fusion blocks."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

def _gn(c):
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0: return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)

def _sat(x):
    x = x[:, :3]  # RGBColorAugment tap may additionally carry saturation.
    mx = x.max(1, keepdim=True).values
    return (mx - x.min(1, keepdim=True).values) / mx.clamp_min(1e-6)

def _stem(out_c):
    c = max(16, min(64, out_c // 4))
    return nn.Sequential(nn.Conv2d(1, c, 3, 2, 1, bias=False), _gn(c), nn.SiLU(),
                         nn.Conv2d(c, out_c, 3, 2, 1, bias=False), _gn(out_c), nn.SiLU())

class SaturationP2Residual(nn.Module):
    def __init__(self, in_channels, gamma_init=0.0):
        super().__init__(); p2, image = in_channels
        if image != 3: raise ValueError("Expected raw RGB tap with 3 channels")
        self.saturation_stem = _stem(p2); self.gamma = nn.Parameter(torch.tensor(float(gamma_init))); self.last_shapes = {}
    def forward(self, inputs):
        p2, image = inputs; s2 = F.interpolate(self.saturation_stem(_sat(image)), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        self.last_shapes = {"P2": tuple(p2.shape), "S2": tuple(s2.shape), "P2_out": tuple(p2.shape)}
        return p2 + self.gamma * s2


class SaturationP2Cue(nn.Module):
    """A single lightweight saturation cue for the final P3→P2 FPN concat.

    It consumes raw RGB only, generates HSV saturation at stride 4, and returns
    ``alpha * S2``.  Unlike the other saturation modules, it neither modifies
    P2/F2 nor uses any semantic feature/context.
    """

    def __init__(self, in_channels: int, out_channels: int = 16, alpha_init: float = 0.1):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"SaturationP2Cue expects raw RGB (3 channels), got {in_channels}")
        out_channels = int(out_channels)
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        self.out_channels = out_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, stride=2, padding=1, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.last_s2 = None
        self.last_effective_s2 = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        s2 = self.stem(_sat(image))
        effective_s2 = self.alpha * s2
        self.last_s2 = s2.detach()
        self.last_effective_s2 = effective_s2.detach()
        return effective_s2

class SaturationGuidedP3Residual(nn.Module):
    def __init__(self, in_channels, embed_channels=32, gamma_init=0.0):
        super().__init__(); f2, p2, f3, image = in_channels
        if image != 3: raise ValueError("Expected raw RGB tap with 3 channels")
        self.p2_proj = nn.Sequential(nn.Conv2d(p2, embed_channels, 1, bias=False), _gn(embed_channels), nn.SiLU()); self.sat_stem = _stem(embed_channels)
        self.sem_proj = nn.Sequential(nn.Conv2d(f3, f2, 1, bias=False), _gn(f2), nn.SiLU())
        self.gate = nn.Sequential(nn.Conv2d(embed_channels * 2, embed_channels, 3, 1, 1, bias=False), _gn(embed_channels), nn.SiLU(), nn.Conv2d(embed_channels, 1, 1))
        self.residual = nn.Sequential(nn.Conv2d(f2, f2, 3, 1, 1, bias=False), _gn(f2), nn.SiLU()); self.gamma = nn.Parameter(torch.tensor(float(gamma_init))); self.last_shapes = {}
    def forward(self, inputs):
        f2, p2, f3, image = inputs; size = f2.shape[-2:]; p2e = self.p2_proj(p2); s2 = F.interpolate(self.sat_stem(_sat(image)), size=size, mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.gate(torch.cat((p2e, s2), 1))); sem = F.interpolate(self.sem_proj(f3), size=size, mode="bilinear", align_corners=False); out = f2 + self.gamma * gate * self.residual(sem)
        self.last_shapes = {"P2": tuple(p2.shape), "F3": tuple(f3.shape), "S2": tuple(s2.shape), "F2_out": tuple(out.shape)}; return out

class SaturationFeatureFilter(nn.Module):
    def __init__(self, in_channels, embed_channels=32, gamma_init=0.0):
        super().__init__(); f2, p2, f3, f4, image = in_channels
        if image != 3: raise ValueError("Expected raw RGB tap with 3 channels")
        self.p2_proj = nn.Sequential(nn.Conv2d(p2, embed_channels, 1, bias=False), _gn(embed_channels), nn.SiLU()); self.sat_stem = _stem(embed_channels)
        self.p3_proj = nn.Conv2d(f3, f2, 1, bias=False); self.p4_proj = nn.Conv2d(f4, f2, 1, bias=False); self.selector = nn.Conv2d(embed_channels, 2, 1)
        self.filter = nn.Sequential(nn.Conv2d(embed_channels + f2, embed_channels, 3, 1, 1, bias=False), _gn(embed_channels), nn.SiLU(), nn.Conv2d(embed_channels, 1, 1)); self.refine = nn.Sequential(nn.Conv2d(f2, f2, 3, 1, 1, bias=False), _gn(f2), nn.SiLU()); self.gamma = nn.Parameter(torch.tensor(float(gamma_init))); self.last_shapes = {}
    def forward(self, inputs):
        f2, p2, f3, f4, image = inputs; size = f2.shape[-2:]; p2e = self.p2_proj(p2); s2 = F.interpolate(self.sat_stem(_sat(image)), size=size, mode="bilinear", align_corners=False); w = torch.softmax(self.selector(p2e + s2), 1)
        c3 = F.interpolate(self.p3_proj(f3), size=size, mode="bilinear", align_corners=False); c4 = F.interpolate(self.p4_proj(f4), size=size, mode="bilinear", align_corners=False); context = w[:, 0:1] * c3 + w[:, 1:2] * c4; mask = torch.sigmoid(self.filter(torch.cat((p2e, context), 1))); out = f2 + self.gamma * mask * self.refine(context)
        self.last_shapes = {"P2": tuple(p2.shape), "F3": tuple(f3.shape), "F4": tuple(f4.shape), "S2": tuple(s2.shape), "F2_out": tuple(out.shape)}; return out

__all__ = ("SaturationP2Residual", "SaturationP2Cue", "SaturationGuidedP3Residual", "SaturationFeatureFilter")


class FilteredP2GuidedP3Residual(nn.Module):
    """Filter P2 only inside a branch, then inject its selected detail into P3.

    The original P2 tensor is never returned or overwritten.  It remains available
    to the final P2 FPN fusion unchanged; this block returns only P3' .
    """
    def __init__(self, in_channels, gamma_init=0.0):
        super().__init__(); p2_c, p3_c = in_channels
        self.filter = nn.Sequential(nn.Conv2d(p2_c, p2_c, 3, 1, 1, groups=p2_c, bias=False), _gn(p2_c), nn.SiLU())
        self.filtered_down = nn.Sequential(nn.Conv2d(p2_c, p2_c, 3, 2, 1, groups=p2_c, bias=False), _gn(p2_c), nn.SiLU(), nn.Conv2d(p2_c, p3_c, 1, bias=False))
        self.base_down = nn.Sequential(nn.Conv2d(p2_c, p2_c, 3, 2, 1, groups=p2_c, bias=False), _gn(p2_c), nn.SiLU(), nn.Conv2d(p2_c, p3_c, 1, bias=False))
        self.gate = nn.Sequential(nn.Conv2d(p3_c * 2, p3_c * 2, 3, 1, 1, groups=p3_c * 2, bias=False), _gn(p3_c * 2), nn.SiLU(), nn.Conv2d(p3_c * 2, 1, 1))
        self.alpha3 = nn.Parameter(torch.tensor(float(gamma_init))); self.last_shapes = {}

    def forward(self, inputs):
        p2, p3 = inputs; filtered = self.filter(p2); d2 = self.filtered_down(filtered); p2d = self.base_down(p2)
        if d2.shape[-2:] != p3.shape[-2:]:
            d2 = F.interpolate(d2, size=p3.shape[-2:], mode="bilinear", align_corners=False); p2d = F.interpolate(p2d, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        mg = torch.sigmoid(self.gate(torch.cat((p2d, p3), 1))); delta = mg * d2; out = p3 + self.alpha3 * delta
        self.last_shapes = {"P2": tuple(p2.shape), "filtered_P2": tuple(filtered.shape), "P2_down": tuple(d2.shape), "P3": tuple(p3.shape), "Mg": tuple(mg.shape), "Delta3": tuple(delta.shape), "P3_out": tuple(out.shape)}
        return out


class DualGateP2P3Residual(nn.Module):
    """Two lightweight P2 reliability/compatibility masks for residual P3 guidance."""
    def __init__(self, in_channels, gamma_init=0.0):
        super().__init__(); p2_c, p3_c = in_channels
        self.filter_gate = nn.Sequential(nn.Conv2d(p2_c, p2_c, 3, 1, 1, groups=p2_c, bias=False), _gn(p2_c), nn.SiLU(), nn.Conv2d(p2_c, 1, 1))
        self.project_down = nn.Sequential(nn.Conv2d(p2_c, p2_c, 3, 2, 1, groups=p2_c, bias=False), _gn(p2_c), nn.SiLU(), nn.Conv2d(p2_c, p3_c, 1, bias=False))
        self.guidance_gate = nn.Sequential(nn.Conv2d(p3_c * 2, p3_c * 2, 3, 1, 1, groups=p3_c * 2, bias=False), _gn(p3_c * 2), nn.SiLU(), nn.Conv2d(p3_c * 2, 1, 1))
        self.alpha3 = nn.Parameter(torch.tensor(float(gamma_init))); self.last_shapes = {}

    def forward(self, inputs):
        p2, p3 = inputs; mf = torch.sigmoid(self.filter_gate(p2)); p2d = self.project_down(p2)
        mf_down = F.interpolate(mf, size=p3.shape[-2:], mode="area")
        if p2d.shape[-2:] != p3.shape[-2:]: p2d = F.interpolate(p2d, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        mg = torch.sigmoid(self.guidance_gate(torch.cat((p2d, p3), 1))); delta = mf_down * mg * p2d; out = p3 + self.alpha3 * delta
        self.last_shapes = {"P2": tuple(p2.shape), "Mf": tuple(mf.shape), "Mf_down": tuple(mf_down.shape), "P2_down": tuple(p2d.shape), "P3": tuple(p3.shape), "Mg": tuple(mg.shape), "Delta3": tuple(delta.shape), "P3_out": tuple(out.shape)}
        return out


__all__ += ("FilteredP2GuidedP3Residual", "DualGateP2P3Residual")


class SaturationStemF2Residual(nn.Module):
    """Post-FPN saturation-only residual; F2 is an exact identity at gamma=0."""
    def __init__(self, in_channels, gamma_init=0.0):
        super().__init__(); f2_c, _ = in_channels
        self.stem = _stem(f2_c)
        self.project = nn.Sequential(nn.Conv2d(f2_c, f2_c, 1, bias=False), _gn(f2_c))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init))); self.last_shapes = {}
    def forward(self, inputs):
        f2, image = inputs; residual = F.interpolate(self.project(self.stem(_sat(image))), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        out = f2 + self.gamma * residual; self.last_shapes = {"F2_base": tuple(f2.shape), "S2": tuple(residual.shape), "residual": tuple(residual.shape), "F2_out": tuple(out.shape)}; return out


class SaturationGuidedP3F2Residual(nn.Module):
    """Post-FPN semantic F3 residual gated by new-branch F2 and saturation cues."""
    def __init__(self, in_channels, embed_channels=32, gamma_init=0.0):
        super().__init__(); f2_c, f3_c, _ = in_channels
        self.f2_proj = nn.Sequential(nn.Conv2d(f2_c, embed_channels, 1, bias=False), _gn(embed_channels), nn.SiLU())
        self.sat_stem = _stem(embed_channels)
        self.f3_proj = nn.Sequential(nn.Conv2d(f3_c, f2_c, 1, bias=False), _gn(f2_c), nn.SiLU())
        self.gate = nn.Sequential(nn.Conv2d(embed_channels * 2, embed_channels, 3, 1, 1, bias=False), _gn(embed_channels), nn.SiLU(), nn.Conv2d(embed_channels, 1, 1))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init))); self.last_shapes = {}
    def forward(self, inputs):
        f2, f3, image = inputs; size=f2.shape[-2:]; f2e=self.f2_proj(f2); s2=F.interpolate(self.sat_stem(_sat(image)), size=size, mode="bilinear", align_corners=False)
        content=F.interpolate(self.f3_proj(f3), size=size, mode="bilinear", align_corners=False); mask=torch.sigmoid(self.gate(torch.cat((f2e,s2),1))); residual=mask*content; out=f2+self.gamma*residual
        self.last_shapes={"F2_base":tuple(f2.shape),"F3":tuple(f3.shape),"S2":tuple(s2.shape),"M2":tuple(mask.shape),"residual":tuple(residual.shape),"F2_out":tuple(out.shape)}; return out


class RegularizedSaturationF2Fusion(nn.Module):
    """Low-capacity post-FPN F3 fusion with GN, depthwise selector and dropout."""
    def __init__(self, in_channels, dropout=0.10, gamma_max=0.5):
        super().__init__(); f2_c, f3_c, _ = in_channels; mid=max(16, f2_c//4)
        self.f2_reduce=nn.Sequential(nn.Conv2d(f2_c,mid,1,bias=False),_gn(mid),nn.SiLU())
        self.f3_reduce=nn.Sequential(nn.Conv2d(f3_c,mid,1,bias=False),_gn(mid),nn.SiLU())
        self.sat_stem=_stem(mid)
        zc=mid*3
        self.mask=nn.Sequential(nn.Conv2d(zc,zc,3,1,1,groups=zc,bias=False),_gn(zc),nn.SiLU(),nn.Dropout2d(dropout),nn.Conv2d(zc,1,1))
        self.content=nn.Sequential(nn.Conv2d(f3_c,f2_c,1,bias=False),_gn(f2_c),nn.SiLU(),nn.Dropout2d(dropout*0.5))
        self.gamma=nn.Parameter(torch.zeros(())); self.gamma_max=float(gamma_max); self.last_shapes={}
    def forward(self, inputs):
        f2,f3,image=inputs; size=f2.shape[-2:]; f2r=self.f2_reduce(f2); f3r=F.interpolate(self.f3_reduce(f3),size=size,mode="bilinear",align_corners=False); s2=F.interpolate(self.sat_stem(_sat(image)),size=size,mode="bilinear",align_corners=False)
        mask=torch.sigmoid(self.mask(torch.cat((f2r,f3r,s2),1))); content=F.interpolate(self.content(f3),size=size,mode="bilinear",align_corners=False); residual=mask*content; out=f2+(self.gamma_max*torch.tanh(self.gamma))*residual
        self.last_shapes={"F2_base":tuple(f2.shape),"F3":tuple(f3.shape),"S2":tuple(s2.shape),"M2":tuple(mask.shape),"residual":tuple(residual.shape),"F2_out":tuple(out.shape)}; return out


__all__ += ("SaturationStemF2Residual", "SaturationGuidedP3F2Residual", "RegularizedSaturationF2Fusion")
