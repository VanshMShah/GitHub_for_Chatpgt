# ATS 3.0

- Add forecast history, performance, models, paper account, system health and settings views.
- Add asynchronous subprocess jobs and SQLite backups; daily experts are persisted and refreshed across processes.
- Repair daily probability calculations, completed-session targeting, outcome settlement, residual input alignment and forecast reuse.
- Isolate one-minute instrument data; enforce instrument identity and checkpoint validation for paper evaluation.
- Correct neural sequence alignment and fit normalizers before validation/holdout; add explicit intraday promotion guards.
- Persist paper orders, fills, costs and positions atomically. Cancel interrupted intents and IOC remainders; preserve positions across sessions.
- Model shared displayed depth and lot-sized partial fills; use source timestamps and position-reducing risk checks.
- Add automatic .env loading, local API protection, Windows setup and an isolated synthetic offline demo.
- Retain live brokerage routing as out of scope. Real authenticated provider operation and profitable forecasting remain unverified.

## Earlier releases

## 2.4.6
- Hostile execution-depth audit found that `NSE_INDEX|Nifty 50` is not an executable order-book instrument; live paper execution now fail-closes on index keys.
- Replaced synthetic BBO execution with genuine Upstox V3 Full Market Quotes depth for configured tradeable `NSE_EQ|` or `NSE_FO|` instruments.
- Paper fills now consume displayed top-5 depth levels with weighted execution price and configurable slippage.
- Live order simulation waits for the configured execution latency, then fetches a fresh exchange-backed depth snapshot before local fill calculation.
- Removed synthetic-BBO configuration knobs from the live runner.
- Added hostile-depth, tradeable-instrument, multi-level-fill, and full-quote normalization tests.
- Verification: 36/36 tests passing; compileall passing; paper replay passing.

## 2.4.5 - Live-data paper runner

- Added NSE market-day/session guard with 2026 exchange holiday calendar.
- Added `LIVE_DATA/PAPER` runner with heartbeat, persistent JSONL audit logs, kill-switch, and local-only execution.
- Added explicit opt-in synthetic BBO mode because the current Upstox adapter does not expose full BBO/depth in this path.
- No broker order endpoint is called by the runner.

# ATS v2 Changelog

## 2.0.0-phase1

- Added persistent local state and recovery-ready schema.
- Added NIFTY + cross-asset historical ingestion adapter.
- Added daily probabilistic expert pool.
- Added 15-minute intraday directional micro-forecast expert.
- Added adaptive DMA-style policy state.
- Added prediction ledger and settlement.
- Added diagnostics and persistent policy history.
- Added GDELT event ingestion adapter.
- Added native asyncio background worker.
- Added cross-platform single-instance lock.
- Added local dashboard.
- Added optional custom Micro-PQRNN training module.
- Added walk-forward validation primitive.

## 2.1.0
- Added Upstox live NIFTY quote adapter with Yahoo fallback.
- Added native auto-start background agent and background-only entrypoint.
- Added risk profiles and forecast-only decision guardrails.
- Added feed-health status and live source visibility.
- Added event ingestion to background cycles.

### Automatic PQRNN lifecycle
- Added evidence-triggered Micro-PQRNN challenger training.
- Added point-in-time sequence construction with train-only normalization.
- Added same-holdout incumbent/challenger comparison and paired bootstrap gate.
- Added persistent champion/challenger artifacts and promotion/rejection diagnostics.
- Added automatic lifecycle execution during off-market sync and `/api/pqrnn` status/manual training endpoints.

## 2.1.1 Automatic resilience loop
- Added bounded exponential-backoff retries for external HTTP data calls.
- Added aggregate request pacing below a nominal 10 calls/sec provider limit.
- Added component-level background retry loops with persisted health state.
- Added PQRNN auto-retry and concurrent-run protection.
- Failed PQRNN updates keep the incumbent and retry on a later cycle.
- Added resilience tests.

## v2.2.0
- Promoted PQRNN champions now load automatically as a fifth daily forecast expert.
- DMA registers PQRNN with a bounded 10% entry prior; subsequent weight is earned from settled outcomes.
- Policy state now exposes `weight_source` so the dashboard distinguishes prior vs learned weights.
- PQRNN inference failures are fail-open: the incumbent ensemble remains active and the failure is persisted.
- Dashboard now shows PQRNN ensemble eligibility and policy weight provenance.

## 2.3.0
- Progressive intraday backfill: 10d -> 20d -> 30d when fallback history is too short.
- Cached intraday history remains usable when the provider fails.
- Intraday health now records the source/range used and required row depth.
- Added redundancy regression test for short-history fallback.

## 2.4.1
- Build tag: `ATS_v2_pqrnn_statistical_repair`.
- Replaced i.i.d. PQRNN promotion-gate bootstrap resampling with configurable moving-block bootstrap (`block_length=5` default).
- Added a deterministic regression test comparing block and i.i.d. bootstrap confidence-interval width on autocorrelated synthetic errors.
- Changed PQRNN challenger checkpoint selection to use an internal validation slice (15% of pre-holdout training data when sufficient rows exist), preserving the untouched promotion holdout.
- Added a deterministic checkpoint-selection regression test.
- Verification: 16/16 tests passing.

## 2.4.3
- Added a horizon-validated PQRNN-to-paper-trading bridge.
- PQRNN forecasts now declare their `1D` horizon explicitly.
- Added signal thresholding, risk-checked market-order creation, and an end-to-end forecast-to-fill replay test.
- The bridge rejects horizon-mismatched forecasts instead of silently using daily PQRNN output for intraday execution.
- Verification: 25/25 tests passing.


## 2.4.2
- Added deterministic paper-trading execution primitives for market-order replay.
- Added post-latency bid/ask execution, conservative partial fills, configurable slippage, quote-staleness checks, and deterministic replay.
- Added paper risk guard for stale data, per-order notional, portfolio notional, and daily-loss limits.
- Added append-only JSONL audit logger and reproducible replay CLI.
- Verification: 22/22 tests passing.


## 2.4.0
- Fixed manual Forecast to follow market/off-market forecast path.
- Fixed Start/Stop lifecycle so stopped agents can be restarted immediately.
- UI buttons now expose action errors and refresh only after completion.
- Exposed agent running state and explicit horizon fallback.

## 2.4.4
- Added a separate leakage-safe intraday Micro-PQRNN lifecycle for approximately 60-second bars.
- Added explicit future-close target construction with target-end timestamps and a purged holdout boundary.
- Added train-only feature scaling and validation-based checkpoint selection for intraday PQRNN.
- Added bar-resolution enforcement so 15-minute data cannot be mislabeled as 60-second evidence.
- Replaced the legacy intraday logistic forecast path with the intraday PQRNN path, while keeping paper execution disabled until qualifying 60-second data is available.
- Added leakage, scaling, resolution, training-artifact, and inference smoke coverage.
- Verification: 28/28 tests passing; compileall passing.
