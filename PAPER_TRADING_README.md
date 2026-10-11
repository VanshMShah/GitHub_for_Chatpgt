> Historical v2 documentation. For v3 behavior, installation and current limitations see README.md and VALIDATION.md.

# ATS Paper-Trading Evaluator

Current scope includes deterministic replay and opt-in live-data/local-execution paper evaluation. No live order routing exists.

## Replay

```bash
python scripts/replay_paper.py \
  --quotes data/paper_quotes_example.csv \
  --orders data/paper_orders_example.csv
```

The execution model applies configured latency before selecting the next quote, executes market orders against bid/ask rather than mid-price, limits fills by displayed size, and applies configurable slippage.

## Tests

```bash
pytest -q
```

The risk guard rejects stale quotes, excessive order/portfolio notional, and daily-loss breaches. The JSONL audit logger is append-only and suitable for deterministic post-mortem replay.


## Live-data / paper-execution mode

This mode consumes live Upstox intraday bars on market days, runs the 60-second PQRNN when trained on the same execution instrument, and executes orders only in the local deterministic simulator. It never calls a broker order endpoint.

The live runner requires a **tradeable NSE instrument** through `ATS_UPSTOX_EXECUTION_KEY`. Do not use `NSE_INDEX|Nifty 50` for execution: the index is a reference/forecast instrument, not an orderable bid/ask book.

Example shape only:

```bash
export ATS_UPSTOX_ACCESS_TOKEN="YOUR_ACCESS_TOKEN"
export ATS_UPSTOX_EXECUTION_KEY="NSE_EQ|<TRADEABLE-INSTRUMENT>"
export ATS_LIVE_PAPER_ENABLE=true
python scripts/run_live_paper.py --once
```

Continuous market-day runner:

```bash
export ATS_UPSTOX_ACCESS_TOKEN="YOUR_ACCESS_TOKEN"
export ATS_UPSTOX_EXECUTION_KEY="NSE_EQ|<TRADEABLE-INSTRUMENT>"
export ATS_LIVE_PAPER_ENABLE=true
python scripts/run_live_paper.py
```

Execution pricing uses genuine Upstox V3 Full Market Quotes top-5 bid/ask depth. The simulator waits for the configured execution latency, requests a fresh quote, then fills across displayed depth levels subject to partial-fill and slippage rules. Synthetic BBO is no longer supported.

**Model alignment rule:** the intraday PQRNN must be trained and evaluated on the same tradeable execution instrument. A NIFTY 50 index model must not be silently applied to an ETF, stock, or futures contract.

Create `data/PAPER_KILL` to halt the evaluator immediately. Remove it to resume.

## Unattended market-day operation

Install the service template at `/etc/systemd/system/ats-live-paper.service`, set `/opt/ats/.env`, and enable it with:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ats-live-paper
sudo systemctl status ats-live-paper
```

The process remains installed continuously but does nothing outside the configured NSE session. The kill switch is `data/PAPER_KILL`. The 2026 calendar is fail-closed; a new year requires an explicit calendar update before the runner will start trading.
