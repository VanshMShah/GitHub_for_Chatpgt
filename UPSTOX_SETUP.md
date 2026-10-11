> Historical v2 documentation. For v3 behavior, installation and current limitations see README.md and VALIDATION.md.

# Live NIFTY feed

ATS uses the current Upstox V3 market-quote LTP API for low-overhead live NIFTY snapshots. Upstox documents V3 LTP with `last_price`, cumulative volume and previous close, and documents `NSE_INDEX|Nifty 50` as the NIFTY 50 instrument key.

PowerShell:

```powershell
$env:ATS_UPSTOX_ACCESS_TOKEN="YOUR_ACCESS_TOKEN"
$env:ATS_UPSTOX_NIFTY_KEY="NSE_INDEX|Nifty 50"
python run.py
```

ATS checks the broker feed first. When it is unavailable or the token is missing, it falls back to the Yahoo Chart adapter and labels that source as a fallback in the dashboard.

The live adapter is intentionally REST-based for the first lightweight release. Upstox's current V3 Market Data Feed can be added later as a WebSocket/protobuf source without changing the forecast/state contracts.

Upstox V3 is now used for NIFTY intraday forecasting when ATS_UPSTOX_ACCESS_TOKEN is set. Without it, Yahoo remains explicitly labelled fallback.

## Genuine depth paper execution

ATS no longer fabricates bid/ask prices. `LIVE_DATA/PAPER` requires a tradeable NSE instrument configured through `ATS_UPSTOX_EXECUTION_KEY` and reads its current top-5 depth from Upstox V3 Full Market Quotes.

Example shape only:

```powershell
$env:ATS_UPSTOX_EXECUTION_KEY="NSE_EQ|<TRADEABLE-INSTRUMENT>"
```

Do not use `NSE_INDEX|Nifty 50` for execution. The index is a forecast/reference instrument, not an orderable book.

The intraday PQRNN must be trained and evaluated on the same execution instrument before its paper-trading results are interpreted.
