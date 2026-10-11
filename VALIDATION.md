# ATS 3.0 validation and limits

Validation date: 11 October 2026. Original input: ATS_V7_genuine_depth_paper_v2.4.6.zip. This release was developed from that supplied source in the selected repository; the original extracted upload was preserved.

## Executed checks

- Original v2 baseline: 36 tests passed.
- Updated suite: 64 tests passed, including subprocess API jobs, completed-session forecast deduplication, settlement once, frozen residual features, instrument/timeframe isolation, normalization boundaries, overnight feature resets, neural challenger training, persistent account recovery, source-time freshness, lot-sized partial fills and shared snapshot depth.
- Fixture-driven integration: one-minute signal → risk checks → post-latency fill → restart without duplicate fill → actual next-bar settlement. This uses controlled data, not a live broker.
- Browser checks in Chromium: all seven views, saved preferences, kill/resume, desktop layout, 390-pixel mobile layout, no horizontal page overflow and no JavaScript errors. Screenshots use isolated synthetic data only.
- Optional dependency behavior: startup and daily training/forecast succeeded with torch imports deliberately blocked.
- Provided CSV replay: `fills=1 position=2 realized_pnl=0.000000`.
- Python compilation and JavaScript syntax checks passed.

The test environment used Python 3.12 on Linux and CPU torch. Windows setup instructions are supplied but were not executed on a Windows host. Upstream NumPy/Pandas and Starlette emitted deprecation warnings; the checks did not disable them.

## Unverified external operations

A read-only Yahoo request failed with `Tunnel connection failed: 403 Forbidden` from this cloud environment's network proxy. Therefore this run does not claim a new real-data 300-case benchmark, functioning live Yahoo ingestion here, or authenticated Upstox paper execution. The user's supplied terminal log independently shows ingestion/training/forecast/replay succeeded on their Windows computer before this update.

No user Upstox credentials were supplied, extracted or embedded. Configure them locally and run preparation/training/paper checks with the exact selected tradeable instrument. A challenger rejection is a valid result and does not establish a software failure.

## Remaining boundaries

- Local development/research application. Internet deployment requires a separately configured and tested HTTPS/authentication reverse proxy, host rules, monitoring and operational review.
- REST snapshot paper simulation, not a continuously observed exchange book. No real broker orders.
- Fee/tax amounts are configurable combined estimates, not a fully implemented exchange tax engine.
- Supplied 2026 holiday calendar needs verification against exchange notices and must be extended explicitly for later years.
- Conservative cross-asset lag is not complete historical publication-time/revision-aware ingestion.
- Repeated challenger trials reuse a holdout. An independent later evaluation is necessary before model-quality conclusions.
- Expected shortfall is not estimated as a forecast statistic. Empirical tail loss is labeled separately and requires enough settled observations.
- A stopped paper runner's marks are last-observed prices. The kill switch blocks exits too; positions persist when an executable close is unavailable.
- External CLI processes are not supervised by the dashboard. Run one paper owner per account; dashboard-managed jobs use workflow locks.

This document reports software checks, not validated forecasting skill, profitability or production readiness.
