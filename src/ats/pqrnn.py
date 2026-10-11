from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class PQRNNConfig:
    features: int = 9
    hidden: int = 24
    layers: int = 1
    quantiles: tuple[float,...] = (0.05,0.25,0.50,0.75,0.95)

class MicroPQRNN(nn.Module):
    """Small probabilistic temporal expert. Train offline; use as a policy-managed expert."""
    def __init__(self, cfg: PQRNNConfig=PQRNNConfig()):
        super().__init__(); self.cfg=cfg
        self.conv=nn.Conv1d(cfg.features,16,kernel_size=3,padding=2,dilation=1)
        self.gru=nn.GRU(16,cfg.hidden,num_layers=cfg.layers,batch_first=True)
        self.ctx=nn.Sequential(nn.LayerNorm(cfg.hidden),nn.Linear(cfg.hidden,cfg.hidden),nn.ReLU())
        self.qbase=nn.Linear(cfg.hidden,1); self.qdelta=nn.Linear(cfg.hidden,len(cfg.quantiles)-1)
        self.direction=nn.Linear(cfg.hidden,1); self.volatility=nn.Linear(cfg.hidden,1)
    def forward(self,x):
        h=torch.relu(self.conv(x.transpose(1,2)).transpose(1,2))[:,:x.size(1),:]
        h,_=self.gru(h); h=self.ctx(h[:,-1])
        q=[self.qbase(h)]
        d=torch.nn.functional.softplus(self.qdelta(h))
        for i in range(d.shape[1]): q.append(q[-1]+d[:,i:i+1])
        return {"quantiles":torch.cat(q,dim=1),"p_up":torch.sigmoid(self.direction(h)).squeeze(1),"volatility":torch.nn.functional.softplus(self.volatility(h)).squeeze(1)}

def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pinball(pred: torch.Tensor, target: torch.Tensor, quantiles: tuple[float,...]) -> torch.Tensor:
    losses=[]
    for i,q in enumerate(quantiles):
        e=target-pred[:,i]; losses.append(torch.maximum(q*e,(q-1)*e).mean())
    return torch.stack(losses).mean()
