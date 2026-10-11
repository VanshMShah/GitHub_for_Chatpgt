> Historical v2 documentation. For v3 behavior, installation and current limitations see README.md and VALIDATION.md.

# ATS v2 Project State

Objective: lightweight persistent NIFTY 50 probabilistic forecasting agent that adapts from settled prediction errors.

Current architecture:
- NIFTY market state + cross-asset feature ingestion
- Point-in-time-style timestamps
- Persistence, Logistic, Quantile GBDT, ARIMA+GJR-GARCH experts
- DMA policy with persistent weights and loss memory
- Prediction ledger + settlement
- Error diagnosis + conservative residual learner
- Native asyncio background agent
- Local dashboard and configurable risk profile
- Upstox adapter ready, fallback feed supported

Next immediate task:
1. Let the native agent settle forecasts and accumulate live outcomes.
2. Let the evidence-gated PQRNN lifecycle run automatically off-market.
3. Promote PQRNN only after out-of-sample statistical gates pass.
4. Verify the learned DMA weights before enabling any later decision/execution layer.
5. Extend the paper-trading evaluator with signal/model adapters only after execution replay tests remain green.
6. Do not connect live brokerage order routing; the current layer is paper/replay only.
7. Live-data paper mode now requires a tradeable NSE execution instrument and genuine Upstox V3 depth; synthetic BBO is removed.

Latest verified build: ATS 2.4.6 genuine-depth paper evaluator
- PQRNN bootstrap gate uses configurable moving-block bootstrap; default block length 5.
- PQRNN checkpoint selection uses an internal validation slice when the pre-holdout training window is large enough.
- Full test suite: 36/36 passing.
- Added deterministic paper-trading execution layer plus market-day live-data/paper mode: post-latency market fills, displayed-depth consumption, slippage, stale-quote checks, risk guard, JSONL audit logging, replay CLI, and fail-closed tradeable-instrument validation.
- 300-case external walk-forward benchmark was not rerun because this environment could not resolve the external historical-data host; no post-repair benchmark result is claimed.

## Automatic PQRNN lifecycle
The backend now evaluates Micro-PQRNN automatically off-market. It does not retrain blindly on every cycle. A run requires a minimum settled-outcome count, a cooldown of new outcomes since the previous PQRNN training, and either recent loss pressure or repeated structural diagnostics. A candidate is trained from daily point-in-time sequences using train-only normalization, scored on an untouched holdout, compared with a freshly walk-forwarded incumbent on the same holdout, and then subjected to a paired bootstrap log-loss gate plus a practical improvement margin and MAE guard. Failed candidates are rejected and retained only as challenger artifacts. Passing candidates become the durable champion. Model weights are stored under `models/` and ignored by git. No live-market retraining occurs.

## PQRNN ensemble integration
A promoted PQRNN champion is automatically loaded as a fifth daily expert. It enters DMA with a bounded 10% prior and earns further weight from settled outcomes. If the champion is absent or inference fails, the incumbent four-expert ensemble remains active. Dashboard policy weights are explicitly labeled as prior or learned.
