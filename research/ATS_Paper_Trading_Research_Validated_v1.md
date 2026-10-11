# ATS Paper-Trading Evaluator: Validated Research Decisions

## Decision status

**Approved direction:** build ATS as an intraday paper-trading model evaluator, not institutional HFT.

**Target operating regime:** event-driven intraday evaluation, with a 1-second to 60-second decision horizon as the engineering target. This is a project classification, not a claim of exchange-colocated HFT capability.

**Execution authority:** local deterministic simulator for research truth; external paper broker only for adapter/session validation.

## Material corrections to the earlier research

### 1. Alpaca API rate limits must not be hard-coded at 200 requests/minute

The current official Alpaca Broker API documentation states that limits are applied at the correspondent/integration level, are not published as one fixed number, and are returned dynamically in `X-RateLimit-*` response headers. Sandbox limits can also differ from production and are not suitable for load testing.

**Decision:** ATS rate limiting must consume live rate-limit headers when available and use configuration as a safety ceiling, not as an asserted broker fact.

Source: https://docs.alpaca.markets/us/docs/broker-api-rate-limits

### 2. Alpaca paper fills are not realistic enough to be the research execution truth

Official paper-trading documentation states that orders are matched against the best available current NBBO, order quantity is not checked against NBBO quantity, and eligible orders can receive partial fills randomly. This makes the service useful for integration testing but insufficient as the sole simulator for queue/depth-sensitive research.

**Decision:** retain a local deterministic execution engine for research evaluation.

Source: https://docs.alpaca.markets/us/v1.4.2/docs/paper-trading

### 3. IBKR paper trading is useful for integration testing but still simulator-bound

IBKR documents that paper trading uses the same simulator employed by its platforms and notes differences from production, including top-of-book simulated fills and unsupported/altered order behaviors.

**Decision:** IBKR may be evaluated as an integration/paper adapter, but its fills must not be treated as ground-truth live execution.

Source: https://www.interactivebrokers.com/campus/glossary-terms/paper-trading-account/

### 4. Current Indian algo-trading rules materially affect any future live-routing path

NSE currently documents a framework for safer retail participation in algorithmic trading, including client direct API/member-front-end categories and registration requirements. NSE also provides official real-time market-data products, including tick-by-tick and Level 1/2/3 technical feeds.

**Decision:** paper-only ATS can remain isolated from live-routing obligations, but any future NSE/BSE live path requires a separate regulatory and broker/API compliance review. Do not infer live deployment eligibility from paper-trading access.

Sources:
- https://www.nseindia.com/static/trade/platform-services-non-neat-decision-support-tools-algorithm-trading
- https://www.nseindia.com/static/market-data/real-time-data-subscription

## Research architecture

```text
Market Data
    -> Timestamp/Integrity Gate
    -> Point-in-Time Feature Builder
    -> PQRNN / Baseline Predictors
    -> Signal Generator
    -> Risk Engine
    -> Deterministic Execution Simulator
    -> Fill / Position Ledger
    -> P&L / Metrics
    -> Immutable Audit Log
```

No trading component may write future information into the forecasting path. The simulator consumes the market state available at simulated order arrival, not the state at signal creation.

## Execution simulator minimum

**Must model:**

- bid/ask spread crossing;
- execution latency;
- partial fills;
- stale quotes;
- rejected/cancelled orders;
- deterministic position accounting;
- conservative passive-order assumptions;
- transaction costs and slippage;
- session boundaries;
- reproducible event replay.

**Do not initially build:**

- Kafka/RabbitMQ;
- distributed inference;
- GPU inference cluster;
- exchange-colocated architecture;
- a full market simulator pretending to reproduce the complete order book when the source data does not contain that information.

### Queue/depth caution

Queue position is only defensible when the available feed contains enough order/trade information to estimate it. A level-1 quote feed cannot prove exact queue priority. Therefore any queue model based only on BBO data must remain explicitly labeled as a simulation assumption.

## Statistical methodology

### Promotion and model selection

The existing PQRNN lifecycle now uses:

- train-only normalization;
- internal validation checkpoint selection;
- untouched holdout comparison;
- moving-block bootstrap for the paired loss gate;
- practical improvement and MAE checks.

The repair is **estimator-validity work**, not evidence of improved trading performance.

### Trading evaluation

Use separate prediction and trading metrics. At minimum:

- MAE;
- directional accuracy;
- Brier/log loss where probabilistic classification applies;
- net P&L after all modeled frictions;
- maximum drawdown;
- turnover;
- average holding time;
- slippage cost;
- fill/rejection rates.

Do not promote a fixed Sharpe threshold such as `>0.5` merely because it is convenient. The threshold must be justified by the strategy, sample length, dependence structure, and decision objective.

### Data snooping

White’s Reality Check and Hansen’s SPA are appropriate tools when many related strategies/models have been searched and the final winner is being evaluated. Their role is to adjust inference for data snooping, not to create profitability evidence from weak data.

Sources:
- White, 2000, *A Reality Check for Data Snooping*, Econometrica.
- Hansen, 2005, *A Test for Superior Predictive Ability*, Journal of Business & Economic Statistics.
- https://www.sciencedirect.com/science/article/pii/S1049007809000682
- https://www.sciencedirect.com/science/article/pii/S0927538X18300775

### Dependent observations

Financial return series are not safely assumed i.i.d. for high-frequency-style evaluation. Block/bootstrap procedures are therefore preferable to naive independent resampling when dependence matters. The repaired PQRNN lifecycle reflects this principle.

The exact block length should remain a calibrated parameter, not a universal truth. The current default of 5 is an engineering starting point, not a proven optimum.

## High-frequency terminology

The system should **not** be described as institutional HFT.

Use:

> **Intraday paper-trading model evaluator**

or, where useful:

> **event-driven intraday trading evaluator**

Reserve “HFT” for contexts where infrastructure, data, execution path, and latency evidence justify the term.

## Baselines

At minimum compare against:

1. no-trade;
2. naive persistence/direction baseline;
3. simple technical baseline appropriate to the chosen instrument;
4. spread-aware random control;
5. existing ATS expert ensemble without PQRNN.

PQRNN must demonstrate incremental value against the existing system, not merely produce positive standalone P&L.

## Mandatory failure tests

Before trusting paper results, inject:

- duplicate market events;
- out-of-order events;
- stale quotes;
- disconnect/reconnect;
- rejected orders;
- partial fills;
- delayed acknowledgements;
- invalid model outputs;
- clock anomalies;
- risk-limit breaches.

A deterministic replay of the same event stream and configuration should produce the same state transitions and metrics.

## Current implementation status

### Completed

- PQRNN block-bootstrap repair.
- PQRNN validation-based checkpoint-selection repair.
- Regression tests for both repairs.
- Full ATS test suite: **16/16 passing**.
- Build version: **2.4.1**.
- Build tag: `ATS_v2_pqrnn_statistical_repair`.

### Not experimentally verified in this environment

The 300-case external NIFTY walk-forward benchmark was not rerun after the repairs because the execution environment could not resolve the external historical-data host. No new post-repair benchmark numbers are therefore claimed.

## Final research decision

Build the paper evaluator around the existing ATS forecasting stack and a local deterministic execution simulator. Keep broker paper APIs outside the research truth loop. Treat all execution frictions and queue assumptions as explicitly parameterized, evidence-labeled mechanisms. Delay any live-order architecture until a separate broker/data/regulatory audit is completed.
