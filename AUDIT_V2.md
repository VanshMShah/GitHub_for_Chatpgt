> Historical v2 documentation. For v3 behavior, installation and current limitations see README.md and VALIDATION.md.

# ATS v2 Build Audit

## Preserved

The legacy ARIMA, GJR-GARCH/Hansen skew-t, VaR/ES and backtesting modules were retained as the econometric kernel.

## Replaced

- single JSON model snapshot as system state
- request-triggered re-estimation architecture
- static /predict response
- legacy dashboard shell

## Added

- persistent local state database
- NIFTY and cross-asset data adapters
- daily and intraday feature pipelines
- prediction ledger
- outcome settlement
- adaptive DMA-style policy
- diagnostics records
- GDELT event ingestion adapter
- native asyncio background agent
- shutdown/restart persistence
- single-instance lock
- local forecasting dashboard
- optional custom Micro-PQRNN module

## Scientific limits

The build has not established real-world NIFTY forecasting accuracy. All claims of accuracy must come from strict point-in-time, walk-forward validation on legitimate historical data.

The Yahoo adapter is a convenience research feed, not a low-latency broker-grade execution feed. Replace it with a legitimate broker/data-provider adapter for production use.

The current container did not have DuckDB or yfinance and could not install them because package downloads are unavailable. The implementation therefore uses Python's SQLite standard library for durable state and a dependency-free HTTP market adapter. The schema remains intentionally simple so a DuckDB backend can be introduced without changing the forecast/policy contracts.
