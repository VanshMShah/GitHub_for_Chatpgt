> Historical v2 documentation. For v3 behavior, installation and current limitations see README.md and VALIDATION.md.

# Hostile Execution-Depth Audit: ATS 2.4.6

## Verdict

**CONDITIONAL GO for LIVE_DATA/PAPER only. NO-GO for live brokerage routing.**

The previous implementation could not honestly claim genuine execution pricing because it used a synthetic spread around LTP. The deeper issue was more fundamental: the configured instrument was `NSE_INDEX|Nifty 50`, which is not an orderable instrument and therefore cannot supply an executable bid/ask book.

## Findings

### F1. Index execution instrument was invalid
- Severity: **Critical**
- Evidence: Upstox full-quote documentation distinguishes instrument keys and exposes depth for tradeable instruments. The NIFTY 50 index is explicitly an index instrument, not an executable security. citeturn668595search0
- Impact: any synthetic BBO replacement would still have been conceptually invalid.
- Repair: runner now rejects `NSE_INDEX|...` for execution and requires `NSE_EQ|...` or `NSE_FO|...`.

### F2. Synthetic BBO was not acceptable for performance claims
- Severity: **Critical**
- Repair: removed synthetic-BBO switches from the live runtime.

### F3. Genuine depth source is now available
- Severity: **Resolved**
- Evidence: Upstox V3 Full Market Quotes provides exchange-backed snapshots with top-5 buy/sell depth including price, quantity, and order count. citeturn668595search0
- Implementation: `UpstoxLiveProvider.fetch_full_quote()` normalizes those levels.

### F4. Depth-aware fill model was previously insufficient
- Severity: **High**
- Repair: market fills now consume displayed levels sequentially and return a weighted average execution price. Partial fills remain possible.

### F5. Current live source is snapshot, not continuous tick stream
- Severity: **Medium**
- Evidence: Upstox also documents a V3 WebSocket `full` feed carrying five market levels. The current implementation uses the REST full-market snapshot because the runner operates on a 60-second decision cycle. citeturn232997view0
- Limitation: this is genuine depth, but it is not equivalent to continuously observed order-book updates.
- Consequence: results must be described as **60-second live-data paper evaluation using exchange-backed depth snapshots**, not tick-level HFT evidence.

### F6. Model/execution instrument mismatch risk
- Severity: **Critical if ignored**
- Repair: live mode requires `ATS_UPSTOX_EXECUTION_KEY`, and the intraday PQRNN must be trained on that same instrument. The system no longer silently applies a NIFTY-index model to another executable security.

### F7. Latency alignment
- Severity: **Resolved for the current paper mode**
- Repair: after signal/order creation, the runner waits for the configured execution latency and fetches a fresh depth snapshot before invoking the local fill simulator.
- Remaining limitation: REST request/response timing is not the same as a persistent exchange feed or colocated execution path.

## Sanity verification

- `pytest -q`: **36/36 passing**
- `compileall`: passing
- Deterministic paper replay: passing
- Multi-level depth fill test: passing
- Full-quote normalization test: passing
- Index-as-execution rejection: passing
- Synthetic-BBO runtime knobs: removed
- Live broker order endpoint: absent from the live paper path

## Recommendation

1. Run `LIVE_DATA/PAPER` only with a tradeable NSE instrument.
2. Train and evaluate the 60-second PQRNN on that exact instrument.
3. Collect live-data paper results across enough market sessions before drawing performance conclusions.
4. Upgrade to the documented Upstox V3 WebSocket `full` feed before making any claim about continuous tick-level or sub-second execution quality. citeturn232997view0
5. Keep broker order routing disabled until a separate regulatory, operational, and live-routing audit is completed.
