"""Optional Micro-PQRNN trainer. Base ATS does not require PyTorch."""
from __future__ import annotations
import argparse
import sys


def main() -> int:
    try:
        import torch
        from src.ats.pqrnn import MicroPQRNN, PQRNNConfig, parameter_count, pinball
    except ImportError:
        print("PyTorch is not installed. Install optional ML dependencies before training PQRNN.")
        print("See requirements-ml.txt")
        return 2

    from src.ats.data import bars_to_frame, YahooChartProvider
    from src.ats.features import SYMBOLS
    import numpy as np

    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--device',default='auto'); args=ap.parse_args()
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu'
    print(f"Loading real NIFTY history; training device={device}")
    # Real point-in-time sequence preparation is intentionally delegated to the research pipeline.
    # This command is a smoke trainer only until walk-forward promotion criteria are met.
    provider=YahooChartProvider(); bars=bars_to_frame(provider.fetch(SYMBOLS['nifty'],period='5y',interval='1d'))
    if len(bars)<250: print('Insufficient NIFTY history'); return 2
    rng=np.random.default_rng(7)
    x=torch.tensor(rng.normal(size=(64,20,9)),dtype=torch.float32,device=device)
    y=torch.tensor(rng.normal(0,.01,size=64),dtype=torch.float32,device=device)
    m=MicroPQRNN(PQRNNConfig()).to(device); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    print(f'parameters={parameter_count(m):,}')
    for epoch in range(args.epochs):
        out=m(x); loss=pinball(out['quantiles'],y,m.cfg.quantiles); opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1)%5==0: print(f'epoch={epoch+1} loss={loss.item():.6f}')
    print('PQRNN smoke training complete. Do not promote until walk-forward evaluator passes.')
    return 0

if __name__=='__main__': sys.exit(main())
