# ATS 3.0 — daily research and paper evaluation

This update repairs forecast lifecycle and instrument alignment, adds durable paper accounting and introduces seven dashboard views. It has **no live broker order routing**. Forecast quality and paper performance must be evaluated separately from software correctness.

## Windows quick start

Extract into a **new folder** rather than overwriting your running v2 installation. Open PowerShell in that folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\.venv\Scripts\python.exe run.py
```

The first command uses a process-scoped execution policy for this local setup script; it does not change the machine's policy. You can also run its commands manually:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env  # only if .env does not already exist
.\.venv\Scripts\python.exe run.py
```

Open the local dashboard at `http://127.0.0.1:8000` on the computer running it. Click **Sync**; the job downloads completed history, settles eligible predictions, trains current experts and saves a daily forecast. The progress strip reports success or the exact failure. No token is needed for Yahoo daily history. If providers refuse access, inspect System health and the worker log; don't interpret an empty screen as a successful forecast.

## Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env  # only if absent
.venv/bin/python run.py
```

## Offline demo

Run `.venv\Scripts\python.exe run.py --demo` to explore a populated dashboard without external feeds. It uses separately generated synthetic observations, `data/demo_state.sqlite3` and `models/demo/`. The banner explicitly labels demo data. It disables live/paper connections and rejects external sync/training jobs. Demo predictions are not evidence of market accuracy. Restart without `--demo` to return to real research state. The 2026 calendar restriction still applies.

## Daily workflow

Daily forecasts require the latest completed supported NSE session and target the next supported session's close. Repeated requests for the same reference and target reuse the existing forecast. Settlement waits until the target plus publication lag and uses an actual closing price. Forecasts awaiting data are distinguished from forecasts whose targets are still pending. The next successful sync retrains stale daily experts.

The dashboard's Start button runs the daily worker; manual jobs require stopping it first. The paper runner and daily worker are mutually exclusive through the dashboard. A shared file lock serializes modifying workflows across processes. API reads remain available during long jobs. Stop requests retain committed records; an in-progress daily worker component finishes before it stops. Manual jobs run in child processes and may be interrupted.

Original POST endpoints remain but now return **202 + job information** rather than the final prediction. Poll `GET /api/jobs/current` for `SUCCEEDED` or `FAILED`. Final records are available from the dashboard and API.

## Existing v2 state

Stop v2 first. Preserve its entire `data/`, `models/` and credential configuration outside the new folder. Copy `data/ats_state.sqlite3` into the new installation if you want old predictions. SQLite migrations are additive; take a backup before migration. Existing historical predictions are retained. Old forecasts do not have full instrument metadata and default to the original index identity.

**Do not copy old neural champions for use in v3.** Instrument identity and sequence alignment have changed. Retrain and pass the new gates. Old intraday rows cannot establish timeframe identity; execution models read the new interval-specific table. Paper positions in v2 were not durable and cannot be reconstructed automatically.

## Paper prerequisites

1. Install optional neural packages: `.venv\Scripts\python.exe -m pip install -r requirements-ml.txt`.
2. Edit `.env` locally. Set the Upstox token and the exact tradeable `ATS_UPSTOX_EXECUTION_KEY`. Use NSE_EQ or NSE_FO, not NSE_INDEX.
3. Set `ATS_LIVE_PAPER_ENABLE=true` only when preparing to evaluate. Set lot size explicitly for derivatives based on verified contract metadata. Set exposure limits appropriate to the instrument; defaults are deliberately small. Equities do not short by default.
4. Configure combined fee/tax estimates through `ATS_PAPER_FEE_BPS` and `ATS_PAPER_FIXED_FEE`; zero means costs excluded. These are estimates, not a complete exchange/broker tax engine.
5. Restart the application. In Models, prepare one-minute history, then train the intraday challenger outside market hours. A rejected candidate is retained for review and cannot execute. A champion must improve Brier score over the training-frequency baseline, meet the MAE guard and have holdout coverage in the configured acceptance range.
6. Start the paper runner from Paper account during a supported session. Inspect heartbeat, positions and orders. It runs locally and never submits broker orders.

Alternatively use `python scripts/run_live_paper.py --once` or continuous mode. Do not simultaneously run a separate CLI paper process and the dashboard-managed paper process; the application does not supervise unrelated externally launched processes.

Orders are immediate-or-cancel: partial-fill remainders are cancelled. The account is committed atomically with fills and fees. Incomplete intents are cancelled on recovery and never resent. Source timestamps are checked for freshness. Depth snapshots are shared across orders within a simulation run, but queue priority, competition, feed loss and market impact remain simplified.

Default policy: attempt to flatten five minutes before close. A failed exit retains the position and is flagged; restart and the next session preserve it. A kill switch halts **all** simulated fills, including exits. Stop retains positions; it does not fabricate a close. The runner marks unrealized P/L when it receives fresh quotes; marks remain last-observed values while stopped.

## Evaluation

Run the tests:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q
```

Neural-specific tests skip explicitly when torch is absent. Install it for full coverage. Use Performance to run a 30-case walk-forward benchmark; the CLI can run a larger evaluation:

```powershell
.\.venv\Scripts\python.exe -m src.ats.research --years 5y --min-train 252 --cases 300 --out research/latest_verified.json
```

This needs external historical feeds. The dashboard benchmark and live outcome scores are separate. No historical JSON shipped in the old project is treated as a new verified result.

Cross-asset daily returns are conservatively lagged one day. This reduces same-date overseas-close leakage but is not a full historical publication-time or revision-aware data warehouse. Repeated challenger attempts reuse a chronological holdout; use a separate later test/paper period before performance conclusions. Pending overlapping forecasts do not count as independent observations. Expected shortfall remains explicitly unavailable; empirical tail loss is a historical statistic, shown only after 100 settled outcomes.

## Backup, deployment and limitations

Use Settings → Create database backup or `python scripts/backup.py`. This uses SQLite's backup API, not a copy of a live WAL file. Preserve model artifacts and `.env` separately; protect credential backups. `data/` runtime files and models are excluded from Git and release ZIPs.

The server binds only to loopback. Cross-origin modifying requests are rejected, and an optional `ATS_API_TOKEN` protects API access. For remote deployment use a separately configured authenticated HTTPS proxy, reviewed host rules and process supervision. This release is a local application, not an Internet-hardened hosted service.

One-minute rolling features reset across gaps and session breaks, so overnight jumps are not treated as one-minute returns.

The supplied exchange holiday calendar supports **2026 only** and needs verification against exchange notices and an update before later years. No automatic guess is made for an unsupported year. External provider access, real Upstox operation, independently calibrated forecasting and profitable trading are not established by passing unit tests.
