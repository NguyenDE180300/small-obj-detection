"""Small, residual-safe paper-inspired guidance blocks for P2-only YOLO ablations."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _gn(c: int) -> int:
    return next(g for g in (8, 4, 2, 1) if c % g == 0)


def _sat(x: torch.Tensor) -> torch.Tensor:
    x = x[:, :3]
    hi, lo = x.amax(1, keepdim=True), x.amin(1, keepdim=True)
    return torch.where(hi > 0, (hi - lo) / hi.clamp_min(torch.finfo(x.dtype).eps), torch.zeros_like(hi))


class SaturationSAGRIFPNP2(nn.Module):
    """Inject a stride-4 saturation detail stem into final F2 using a P2/F3 gate."""
    def __init__(self, channels: list[int], width: int = 16) -> None:
        super().__init__()
        f2c, p2c, f3c, rawc = channels
        if rawc != 3:
            raise ValueError("SaturationSAGRIFPNP2 requires raw RGB tap")
        self.stem = nn.Sequential(nn.Conv2d(1, width, 3, 2, 1, bias=False), nn.GroupNorm(_gn(width), width), nn.SiLU(), nn.Conv2d(width, width, 3, 2, 1, bias=False), nn.GroupNorm(_gn(width), width), nn.SiLU())
        self.detail = nn.Conv2d(width, f2c, 1, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(p2c + f3c, width, 1, bias=False), nn.GroupNorm(_gn(width), width), nn.SiLU(), nn.Conv2d(width, 1, 3, 1, 1))
        self.gamma_sat = nn.Parameter(torch.zeros(()))
        self.last_gate = self.last_gamma = None

    def forward(self, xs):
        f2, p2, f3_up, rgb = xs
        if f3_up.shape[-2:] != p2.shape[-2:]: f3_up = F.interpolate(f3_up, size=p2.shape[-2:], mode="nearest")
        gate = torch.sigmoid(self.gate(torch.cat((p2, f3_up), 1)))
        sat = self.stem(_sat(rgb))
        sat = F.interpolate(sat, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        out = f2 + self.gamma_sat * gate * self.detail(sat)
        self.last_gate, self.last_gamma = gate.detach(), self.gamma_sat.detach()
        return out


class ResidualMSSEnSimAM(nn.Module):
    """MSS-YOLO global/local-Scharr attention with zero-initialized residual scale."""
    def __init__(self, c: int) -> None:
        super().__init__()
        self.local_weights = nn.Parameter(torch.ones(3))
        self.branch_weights = nn.Parameter(torch.ones(3))
        self.beta_mss = nn.Parameter(torch.zeros(()))
        self.register_buffer("sx", torch.tensor([[-3.,0.,3.],[-10.,0.,10.],[-3.,0.,3.]]).view(1,1,3,3), persistent=False)
        self.register_buffer("sy", torch.tensor([[-3.,-10.,-3.],[0.,0.,0.],[3.,10.,3.]]).view(1,1,3,3), persistent=False)
        self.last_maps = {}

    def forward(self, x):
        mean = x.mean((2,3), keepdim=True); var = (x-mean).square().mean((2,3), keepdim=True)
        ag = torch.sigmoid(1 / ((x-mean).square() / (4*(var+1e-4)) + .5))
        lv = sum(w * F.avg_pool2d((x-F.avg_pool2d(x,k,1,k//2)).square(), k, 1, k//2) for w,k in zip(torch.softmax(self.local_weights,0), (3,5,7)))
        al = torch.sigmoid(lv)
        sx, sy = self.sx.to(x).repeat(x.shape[1],1,1,1), self.sy.to(x).repeat(x.shape[1],1,1,1)
        ae = torch.sigmoid(torch.sqrt(F.conv2d(x,sx,padding=1,groups=x.shape[1]).square()+F.conv2d(x,sy,padding=1,groups=x.shape[1]).square()+1e-6))
        am = sum(w*a for w,a in zip(torch.softmax(self.branch_weights,0),(ag,al,ae)))
        self.last_maps = {"A_global":ag.detach(), "A_local":al.detach(), "A_edge":ae.detach(), "A_mss":am.detach(), "beta":self.beta_mss.detach()}
        return x + self.beta_mss * am * x


class SRMSemanticGate(nn.Module):
    """SRM-derived mask checked by lightweight F3/F4 semantic context; cls path only."""
    def __init__(self, channels: list[int], hidden: int = 16) -> None:
        super().__init__()
        from .srm_f2_guidance import SRMF2Guidance
        f2, f3, f4, raw = channels
        self.srm = SRMF2Guidance([f2, raw], "srm3", 32)
        self.context = nn.Sequential(nn.Conv2d(f3+f4, hidden, 1, bias=False), nn.GroupNorm(_gn(hidden),hidden), nn.SiLU(), nn.Conv2d(hidden,1,1))
        self.fuse = nn.Conv2d(2,1,3,1,1)
        self.last_maps = {}
    def forward(self, xs):
        f2,f3,f4,rgb=xs
        a_srm=self.srm.make_mask(f2, rgb)
        ctx=torch.cat((F.interpolate(f3,size=f2.shape[-2:],mode="bilinear",align_corners=False),F.interpolate(f4,size=f2.shape[-2:],mode="bilinear",align_corners=False)),1)
        semantic=torch.sigmoid(self.context(ctx)); refined=torch.sigmoid(self.fuse(torch.cat((a_srm,semantic),1)))
        gamma=self.srm.effective_gamma(); self.last_maps={"A_srm":a_srm.detach(),"semantic_context":semantic.detach(),"A_semantic":refined.detach(),"gamma":gamma.detach()}
        return f2 + gamma*refined*f2


class SRMEdgeRefine(nn.Module):
    """Refine the SRM classification mask with Scharr and multi-scale local variance."""
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        from .srm_f2_guidance import SRMF2Guidance
        f2, raw = channels; self.srm=SRMF2Guidance([f2,raw],"srm3",32); self.scale_weights=nn.Parameter(torch.ones(3)); self.fuse=nn.Conv2d(3,1,3,1,1)
        self.register_buffer("sx",torch.tensor([[-3.,0.,3.],[-10.,0.,10.],[-3.,0.,3.]]).view(1,1,3,3),persistent=False); self.register_buffer("sy",torch.tensor([[-3.,-10.,-3.],[0.,0.,0.],[3.,10.,3.]]).view(1,1,3,3),persistent=False); self.last_maps={}
    def forward(self,xs):
        f2,rgb=xs; a=self.srm.make_mask(f2, rgb); c=f2.shape[1]; sx=self.sx.to(f2).repeat(c,1,1,1); sy=self.sy.to(f2).repeat(c,1,1,1)
        edge=torch.sigmoid(torch.sqrt(F.conv2d(f2,sx,padding=1,groups=c).square()+F.conv2d(f2,sy,padding=1,groups=c).square()+1e-6).mean(1,keepdim=True))
        local=sum(w*F.avg_pool2d((f2-F.avg_pool2d(f2,k,1,k//2)).square(),k,1,k//2).mean(1,keepdim=True) for w,k in zip(torch.softmax(self.scale_weights,0),(3,5,7))); local=torch.sigmoid(local)
        refined=torch.sigmoid(self.fuse(torch.cat((a,edge,local),1))); gamma=self.srm.effective_gamma(); self.last_maps={"A_srm":a.detach(),"A_edge":edge.detach(),"A_local":local.detach(),"A_refined":refined.detach(),"gamma":gamma.detach()}; return f2+gamma*refined*f2

__all__=("SaturationSAGRIFPNP2","ResidualMSSEnSimAM","SRMSemanticGate","SRMEdgeRefine")
