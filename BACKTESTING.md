# TraderWise — Backtesting Framework

> **Scope:** multi-asset (US equities + commodity futures) swing-trading backtester for the system specified in `final_requirements.txt` and re-targeted in `IMPLEMENTATION_PLAN.md`.
> **Companion doc:** `LOCAL_DEV.md` — what of this runs on a MacBook M3 Pro 18GB.

The single most important architectural rule: **the backtester runs the same scenario-manager code, the same feature pipeline, the same risk gateway, and writes to the same `sessions` table as live trading. Only the `mode` column differs.** Without this, your backtest results do not predict your live results, and every other piece of this document is wasted effort.

---

## 1. What "backtest" means here

Three different things people call "backtest." Each needs a different tool.

| Layer | Question it answers | Tool | Speed |
|---|---|---|---|
| **L1 — Research backtest** | Does this *idea* have edge in principle? | `vectorbt` (vectorized over numpy/pandas) | Seconds–minutes |
| **L2 — Strategy backtest** | Does this *strategy* (signals + sizing + risk) make money after costs? | Custom event-driven engine | Minutes–hours |
| **L3 — System backtest** | Does the *full system* (scenario manager + LLM orchestrator-equivalent + HITL policy + execution) make money? | Same event-driven engine, replaying through `ScenarioManager.on_candle()` | Hours |

Use L1 for fast iteration during research. Promote ideas to L2 once you have a candidate strategy. Promote to L3 only when L2 shows positive expectancy after frictions and you want to validate the orchestration layer end-to-end.

L3 is the level that gives you research/production parity. L1 and L2 are research tools; L3 is the contract with reality.

**The LLM is never invoked during backtest.** Spec §8.6.4 is explicit. Every orchestrator decision in L3 is made by a deterministic policy that mimics what the LLM is allowed to do (pick from a small set of validated structures, deterministic sizing). The LLM is invoked only in live/paper modes. This keeps backtests cheap, reproducible, and free of the "LLM nondeterminism" trap.

---

## 2. Architecture

a```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BACKTEST ENGINE (L2/L3)                         │
│                                                                         │
│  ┌────────────────┐    ┌──────────────────┐    ┌───────────────────┐    │
│  │ DATA REPLAYER  │───▶│ FEATURE PIPELINE │───▶│ MODELS A / B′ /   │    │
│  │  (point-in-    │    │  (same as live)  │    │  C / D inference  │    │
│  │   time loader) │    └──────────────────┘    └────────┬──────────┘    │
│  └────────────────┘                                     │               │
│         │                                               ▼               │
│         │                                  ┌────────────────────────┐   │
│         │                                  │ SCENARIO MANAGER       │   │
│         └─────────────────────────────────▶│  (same code as live)   │   │
│                                            │  on_candle(market)     │   │
│                                            └──────────┬─────────────┘   │
│                                                       │                 │
│                                       scenario_confirmed event          │
│                                                       │                 │
│                                                       ▼                 │
│                                          ┌──────────────────────────┐   │
│                                          │ DETERMINISTIC POLICY     │   │
│                                          │ (LLM stand-in for L3)    │   │
│                                          │   structure + sizing     │   │
│                                          └──────────┬───────────────┘   │
│                                                     │                   │
│                                                     ▼                   │
│                                          ┌──────────────────────────┐   │
│                                          │ RISK GATEWAY             │   │
│                                          │  (same code as live)     │   │
│                                          └──────────┬───────────────┘   │
│                                                     │                   │
│                                                     ▼                   │
│                                          ┌──────────────────────────┐   │
│                                          │ FILL ENGINE              │   │
│                                          │  + slippage + commission │   │
│                                          │  + corporate actions     │   │
│                                          └──────────┬───────────────┘   │
│                                                     │                   │
│                          ┌──────────────────────────┴────────────────┐  │
│                          ▼                                           ▼  │
│             ┌───────────────────┐                      ┌────────────────┐
│             │  SESSIONS TABLE   │                      │  TRADES + P&L  │
│             │  (mode='backtest')│                      │  (mode='back-  │
│             └───────────────────┘                      │   test')       │
│                                                        └────────────────┘
└─────────────────────────────────────────────────────────────────────────┘
```

Same boxes as the live system. The only differences:
- The DATA REPLAYER swaps the WebSocket feed for a chronologically-walked database cursor.
- The FILL ENGINE simulates execution (live uses Alpaca/IBKR adapters).
- The DETERMINISTIC POLICY stands in for the LLM orchestrator at L3.
- All writes are stamped `mode='backtest'`.

Everything in between is the *same Python objects* loaded from the *same packages*.

---

## 3. Folder layout

```
app/backtest/
├── __init__.py
├── engine.py                    # event loop: walk timestamps, dispatch candles
├── data/
│   ├── replayer.py              # asof-correct OHLCV iteration
│   ├── universe.py              # point-in-time S&P 500 / Russell 1000 membership
│   ├── corporate_actions.py     # splits, dividends, mergers, delistings
│   └── futures_roll.py          # deterministic continuous-contract roll
├── frictions/
│   ├── slippage.py              # bps + spread-fraction models per asset class
│   ├── commissions.py           # per-share / per-contract schedules
│   ├── fills.py                 # marketable-limit, cross-the-spread, partial fills
│   └── borrow.py                # short borrow rates + locate availability
├── policies/
│   └── deterministic_policy.py  # LLM stand-in for L3: maps scenario → structure + size
├── labeling/
│   ├── triple_barrier.py        # spec §15.1
│   └── meta_label_inputs.py     # builds (primary_signal, regime) tuples
├── validation/
│   ├── walkforward.py           # quarterly retrains
│   ├── purged_kfold.py          # combinatorial purged CV with embargo
│   └── deflated_sharpe.py       # selection-bias-corrected Sharpe
├── analytics/
│   ├── tearsheet.py             # full report: Sharpe, Sortino, Calmar, MDD, etc.
│   ├── attribution.py           # P&L by scenario_type, by sector, by regime
│   └── drift.py                 # PSI on features, Brier on calibration
├── policies/
│   └── deterministic_policy.py
├── runners/
│   ├── run_l1_research.py       # vectorbt-based fast iteration
│   ├── run_l2_strategy.py       # event-driven, single strategy
│   └── run_l3_system.py         # full scenario-manager replay
└── tests/
    ├── test_no_lookahead.py     # asserts every feature is t-1 only
    ├── test_replay_determinism.py # same seed, same input → same output
    └── test_parity.py           # live and backtest produce identical sessions
                                 # for a fixed historical day fed both ways
```

The crucial test is `tests/test_parity.py`. If you cannot prove identical session records for a fixed historical day fed through both pipelines, you do not have parity. Build that test before scaling up.

---

## 4. Data layer (point-in-time correctness)

This is where most retail backtests silently break. Five rules, none optional:

### 4.1 As-of timestamps on every feature
Every row in the feature table carries `(symbol, available_at, value)`. The backtester at simulation-time `t` sees only rows where `available_at <= t`. This includes:
- **Daily bars** — available at next session open (T+0 close → available for T+1 decisions). Never use today's close to predict today's close.
- **Macro releases** — FOMC statement timestamps are the *announcement* timestamp, not the meeting date. CPI/NFP same.
- **Earnings** — surprise is available on the print timestamp; revisions become available on broker-update timestamps.
- **News sentiment** — available at the article publish timestamp; not the date stamp.

Add a helper:

```python
# app/core/ai/features/asof.py
def asof_join(features_df, decision_times, on='symbol'):
    """Backward-join features to decision times. Forbids future data."""
    return pd.merge_asof(
        decision_times.sort_values('decision_time'),
        features_df.sort_values('available_at'),
        left_on='decision_time', right_on='available_at',
        by=on, direction='backward', allow_exact_matches=False,
    )
```

`allow_exact_matches=False` is deliberate — a feature timestamped at exactly `t` should not be visible at decision-time `t`, only strictly before.

### 4.2 Survivorship-bias-safe universe
Maintain a `universe_membership(symbol, joined_at, left_at)` table. Backtest universe at date `d` is the set of symbols where `joined_at <= d AND (left_at IS NULL OR left_at > d)`. Without this, you train on tomorrow's S&P 500 (which by construction has outperformed) and your backtest beats reality by 3–5% annualized.

For S&P 500 historical membership: the iShares SPY constituents file has clean monthly history; for Russell 1000, FTSE Russell publishes annual reconstitution lists. Reconstruct daily membership by carrying month-end snapshots forward.

### 4.3 Corporate actions
Splits, dividends, mergers, spin-offs, delistings. Adjust *prices*, do not adjust *features that depend on price level* (RSI levels, moving averages) — recompute those from adjusted prices at backtest time.

Delisted names are the most common forgotten case. A delisted stock often dropped 80% before delisting; if your backtest just removes it from the universe, you've quietly survivorship-biased again. Hold the position through the delisting and mark the loss.

### 4.4 Continuous futures contracts
Pick *one* roll rule and apply it deterministically:
- **Open-interest based** (most common): roll N days before front-month expiry, weighted by open interest.
- **Time-based**: roll on the 25th of each contract month.
- **Calendar-based** (per CME): roll on first notice day.

Whichever you pick, document it and never change it without re-backtesting from scratch. The "back-adjusted" continuous series uses gap-adjustment so that returns are continuous; the unadjusted series gives correct *prices* for option-style payoffs (irrelevant for us). Use back-adjusted for swing strategies.

### 4.5 Trading-day calendar
Use `exchange_calendars` (or `pandas_market_calendars`) for NYSE / CME schedules. Half-days, holidays, FOMC half-day-after-Thanksgiving — handle them. Iterating on a naive `pd.date_range(freq='B')` will silently produce trades on Thanksgiving Friday at the wrong session length.

---

## 5. Frictions model

The single biggest source of "backtest-to-live divergence." Get this realistic or backtests are fiction.

### 5.1 Slippage (per asset class)

| Asset class | Default slippage | Notes |
|---|---|---|
| US large-cap equity (>$10B mcap, ADV > 5M shares) | `mid + 0.5 × spread × side` (cross half the spread) | Liquid; tight |
| US mid-cap equity ($2–10B mcap) | `mid + 0.7 × spread × side` | Wider spread, more impact |
| US small-cap equity (<$2B mcap) | `mid + 1.0 × spread × side` + 5 bps impact | Default to skipping for swing trading |
| Liquid ETFs (SPY, QQQ, IWM, sector SPDRs) | `mid + 0.3 × spread × side` | Tight, deep |
| Front-month CL, GC, SI, NG futures | 0.5 ticks past mid | Liquid |
| Back-month or ag futures (ZC, ZS, ZW) | 1.0 tick past mid | Wider |

Add **size impact**: above a threshold of `position_size / ADV > 0.01` (i.e., > 1% of average daily volume), apply Almgren-Chriss-style impact:

```python
impact_bps = c1 * sqrt(participation_rate) + c2 * volatility * sqrt(time)
```

For swing horizon, the temporary impact dominates. Default `c1 = 10` bps, `c2 = 5` bps; calibrate against your actual fills once you have live data.

### 5.2 Commissions

| Venue | Schedule |
|---|---|
| Alpaca (US equities) | $0 commission, $0 per share. Add SEC + TAF fees on sells (~$0.0002/share). |
| IBKR Pro (US equities) | $0.0035/share, min $0.35, max 1% of trade value. Add exchange + clearing fees. |
| IBKR (CME futures) | ~$1.20/contract round-trip including exchange fees. |
| Margin interest (IBKR) | benchmark + 1.5% on tier 1; daily accrual on borrowed cash. |

### 5.3 Fill model
For limit orders (which is what the system places — never market):
- **Order placed at calculated price.** If the market trades through your limit by ≥ 1 tick, you're filled at your limit. If not, you're not filled.
- **30-second escalation:** unfilled → cancel-and-replace at mid + 1 tick (per spec §15.9). Track partial fills for size > top-of-book.
- **End-of-bar resolution:** for daily-bar backtests, an order placed at decision-time `t` resolves against the next bar's intraday range:
    - If `next_bar.low <= limit_price <= next_bar.high`, filled at limit (with slippage).
    - Else not filled. The order does *not* persist to the bar after.
- **Conservative bias:** for borderline cases (limit equals next-bar high or low), assume *no fill*. This biases toward realism.

### 5.4 Short-side frictions
Equities only:
- **Borrow rate** by tier: easy-to-borrow names ~0.25%/yr; general collateral 0.5%; hard-to-borrow can be 5–50%/yr. Use IBKR's stock loan rates as ground truth; default to 0.5%/yr if unknown.
- **Locate availability:** maintain a `borrow_availability(symbol, date, available_qty)` table; reject shorts where supply was insufficient on the historical date.
- **Hard-to-borrow names** should be excluded from the short universe by default — paying 20%/yr borrow on a 5-day swing eats 27 bps per trade.

### 5.5 Margin & financing
- **Reg T:** 50% initial margin, 25% maintenance.
- **Portfolio margin** (IBKR, $110K minimum): more efficient but harder to model. Default to Reg T for backtests until live PM is enabled.
- **Daily mark-to-market** for futures with overnight margin requirements per CME SPAN. Approximation: 5–10% of notional, depending on asset.
- **Cash interest** on idle cash: T-bill rate as a proxy. Negligible for short backtests; matters for multi-year ones.

### 5.6 Failures to model (worth knowing about, not worth modeling at v1)
- Latency between signal and order arrival at exchange.
- Order-book queue position.
- Iceberg order behavior.
- Cross-venue routing.

Skip these until they prove to matter against live data. Realistic spread + impact + commission gets you ~80% of live realism for swing trading.

---

## 6. Labeling and validation methodology

This is where most "Sharpe 3.0 backtests" come from — bad labels and bad CV.

### 6.1 Triple-barrier labeling (spec §15.1)
For every candidate entry signal at time `t`, define three barriers:
- **Profit barrier:** entry × (1 + k_pt × ATR_pct)
- **Stop barrier:** entry × (1 − k_sl × ATR_pct)
- **Time barrier:** `t + N` trading days

Walk forward bar-by-bar; the label is whichever barrier is hit first:
- `+1` if profit barrier hits first
- `−1` if stop barrier hits first
- `0` if time barrier hits first (no decisive move)

Defaults: `k_pt = 2.0`, `k_sl = 1.0` (asymmetric, encodes the system's R:R target), `N = 10` for swing.

This labels the model on **what actually generates P&L** (path-dependent outcomes), not on a naive "did price end up?" question. Implementation: `mlfinlab.labeling.triple_barrier_labels` is the canonical reference; in practice, write your own ~80-line vectorized version using `numba` for speed (the `mlfinlab` version is correct but slow on universes).

### 6.2 Meta-labeling (spec §15.2)
Two-stage:
1. **Primary model** (Models A/B′/C): predicts a tradeable signal (e.g., "vol expansion likely").
2. **Meta model**: takes the primary signal as a feature plus regime features and outputs `P(signal will pay off)`. Trade only when `P > threshold` (typically 0.55–0.6).

Train the meta model on **out-of-fold** primary predictions to avoid leakage. The meta model's labels come from triple-barrier outcomes of the primary signal.

### 6.3 Combinatorial purged k-fold CV (spec §15.4)
Standard k-fold leaks because adjacent rows in time series are autocorrelated. Combinatorial purged CV:
- Split time into N blocks (default N=10 for ~5 years of daily data).
- Use combinations of blocks for train/test (not just sequential).
- **Purge** training observations whose label window overlaps with the test set.
- **Embargo** a buffer (default 5 trading days) between train and test boundaries.

This produces a *distribution* of out-of-sample Sharpe, not a single number. Decision rule: **if 5th-percentile Sharpe > 0.5, the strategy probably has edge. If only the median is positive, you got lucky in selection.**

### 6.4 Walk-forward refitting (spec §5)
For production-style validation:
- Train on 2018–2022, validate on 2023, test on 2024.
- Roll quarterly: at quarter-end, re-fit on the new training window, freeze, evaluate the next quarter as out-of-sample.
- This is what the live system will do; backtests should mirror it.

### 6.5 Deflated Sharpe ratio (spec §15.5)
When you've tried K strategies and pick the best, the winner's reported Sharpe is biased upward by selection. Deflated Sharpe corrects for:
- Number of trials K.
- Skew and kurtosis of returns.
- Track length.

Reference: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio." Implementation: `mlfinlab.backtest_statistics.bt_statistics`. Rule of thumb: if deflated Sharpe < 1.0 from a Sharpe-2.0 backtest, you likely overfit during selection — pause and reformulate.

### 6.6 No-look-ahead audit (mechanical, not optional)
A test that runs in CI:

```python
def test_no_lookahead():
    """For every feature, decision-time t can only see data with available_at < t."""
    features = load_all_feature_definitions()
    for f in features:
        decision_times = sample_decision_times(n=1000)
        for t in decision_times:
            visible = f.compute(asof=t)
            assert visible.available_at.max() < t, \
                f"Leakage in {f.name}: saw data at {visible.available_at.max()} for decision at {t}"
```

This catches 90% of accidental leakage. Run it on every PR.

---

## 7. The deterministic L3 policy (LLM stand-in)

For the L3 system backtest, the LLM is replaced by a deterministic policy that simulates what the LLM would be allowed to do. This keeps backtests cheap and reproducible.

**Why this is acceptable:** the LLM in this system never makes risk-bearing decisions on its own — it picks from a small set of validated structures, with sizing computed deterministically. A policy that mechanically picks the same structure for the same scenario type and computes the same size will, on average, produce the same expectancy as the LLM, minus some narrative quality.

```python
# app/backtest/policies/deterministic_policy.py
class DeterministicPolicy:
    """Stand-in for the LLM orchestrator during backtest."""

    structure_map = {
        ("trend_continuation", "high_vol"): "long_stock_atr_stop",
        ("trend_continuation", "mid_vol"):  "long_stock_atr_stop",
        ("trend_continuation", "low_vol"):  "long_stock_tight_stop",
        ("mean_reversion",     "any"):      "long_stock_atr_stop",
        ("breakout",           "any"):      "long_stock_atr_stop",
        ("vol_expansion",      "any"):      "reduce_existing_long",
        ("vol_crush",          "any"):      "long_spy_or_qqq",
        ("pead",               "any"):      "long_stock_5d_hold",
        ("term_structure_flip","any"):      "long_front_short_back_futures_pair",
    }

    def decide(self, scenario_event, market_state, calibrated_p):
        regime = classify_regime(market_state)
        struct = self.structure_map[(scenario_event.scenario_type, regime)]
        size = kelly_size(p=calibrated_p, R=2.0, fraction=0.25, cap=0.01)
        return TradePlan(structure=struct, size=size, ...)
```

When you run live, swap this for the LLM call. The orchestrator must be constrained to the same structure_map (enforced by the `propose_trade` tool's accepted-structure list) to preserve parity.

---

## 8. Engine internals

### 8.1 Event loop (single-threaded, deterministic)

```python
# app/backtest/engine.py
class BacktestEngine:
    def __init__(self, start, end, universe, scenario_manager, policy, risk, fills):
        self.replayer = DataReplayer(start, end, universe)
        self.scenarios = scenario_manager     # SAME object as live uses
        self.policy = policy
        self.risk = risk                       # SAME risk gateway as live
        self.fills = fills
        self.portfolio = Portfolio()

    def run(self):
        for ts, market_snapshot in self.replayer:
            # 1. Update open positions: marks, stops, time-exits
            self.portfolio.mark(ts, market_snapshot)
            for trade in self.portfolio.open_trades():
                if trade.kill_criteria_hit(market_snapshot):
                    self.fills.exit(trade, market_snapshot)

            # 2. Feed the candle to the scenario manager (numerical update)
            events = self.scenarios.on_candle(market_snapshot)

            # 3. For each confirmed scenario, ask the policy and try to fill
            for ev in events:
                if ev.kind != 'confirmed':
                    continue
                plan = self.policy.decide(ev, market_snapshot, ev.calibrated_p)
                approved = self.risk.validate(plan, self.portfolio, market_snapshot)
                if approved:
                    self.fills.enter(approved, market_snapshot)

            # 4. End-of-bar bookkeeping
            self.portfolio.snapshot(ts)

        return self.portfolio.tearsheet()
```

This is ~50 lines of essence. The `scenarios`, `risk`, and `policy.decide → TradePlan` shapes are identical to live; only `replayer` and `fills` are simulator-specific.

### 8.2 Determinism

Three sources of nondeterminism to lock down:
- **Random seeds:** every model has a `random_state`; every Monte Carlo run threads the same seed.
- **Floating-point order:** sort all merges and groupbys by stable keys before reductions.
- **Library versions:** pin every dep in `requirements-backtest.txt`. A LightGBM minor-version change has been known to flip predictions by epsilon; that epsilon flips a borderline trade; your Sharpe moves.

The parity test should pass byte-for-byte across runs and across machines.

### 8.3 Performance budgets
| Backtest type | Target wall time on M3 Pro 18GB |
|---|---|
| L1 single-asset 5-year vectorbt | < 5 seconds |
| L2 single-asset 5-year event-driven | < 30 seconds |
| L2 200-name S&P universe 5-year event-driven | < 10 minutes |
| L3 200-name 5-year full system | < 30 minutes |
| Walk-forward (20 quarters × full backtest) | < 4 hours overnight |
| Combinatorial purged CV (50 train/test combos) | < 6 hours overnight |

If any of these blow out, profile before optimizing. Most slowdowns are I/O (re-loading data per run) or pandas operations on the hot path; switching to numpy arrays + numba inside the inner loop typically fixes it.

---

## 9. Tearsheet and analytics

Every backtest produces a structured report:

```
================================================================
BACKTEST REPORT — strategy_v3.7
period: 2018-01-02 → 2025-12-31    universe: SP500 PIT
mode: l3_full_system                seed: 42
================================================================

P&L:
  total return                  +147.2%
  CAGR                          +13.8%
  Sharpe (raw)                  1.42
  Sharpe (deflated, K=18)       1.05
  Sortino                       1.93
  Calmar                        1.15
  max drawdown                  −12.0%   (2022-09 to 2023-01)
  time underwater (max)         87 days
  % winning months              63%
  tail ratio (P95/|P5|)         1.34

TRADES:
  total                         412
  win rate                      52.4%
  avg win                       +1.81%
  avg loss                      −0.94%
  R:R                           1.93
  avg holding period            7.2 days
  largest win                   +6.8%
  largest loss                  −2.1%
  consecutive losses (max)      6

ATTRIBUTION:
  by scenario_type:
    trend_continuation          +62.4%   201 trades  win 54%
    mean_reversion              +28.1%    98 trades  win 51%
    breakout                    +33.7%    71 trades  win 49%
    vol_expansion               +12.1%    18 trades  win 67%
    pead                        +10.9%    24 trades  win 58%
  by sector (top 5):            ...
  by regime (low/mid/high vol): ...

DRIFT:
  feature drift PSI (max over features)    0.18  ⚠ above 0.15 in 2024Q3
  calibration drift (60-day rolling Brier) stable

VALIDATION:
  combinatorial purged CV    median Sharpe 1.18   5%ile Sharpe 0.62  ✓
  walk-forward (20 quarters) median Sharpe 1.31   worst quarter Sharpe -0.34
  deflated Sharpe            1.05  (Sharpe 1.42 over 7y, K=18 strategies tried)

FRICTIONS:
  total commissions            $1,247
  total slippage estimate      $4,892
  total borrow cost            $312
  P&L gross of frictions       +156.4%
  P&L net of frictions         +147.2%
  cost drag                    9.2 pp / year × 0.07 = 0.6 pp
================================================================
```

Implementation: `app/backtest/analytics/tearsheet.py` produces both the text report and a JSON artifact written next to the run output. Grafana can read the JSON and chart strategy comparison; the text version is for code review.

---

## 10. Multi-asset / portfolio simulation

For an equity universe of ~200 names, the engine must handle:

- **Cross-sectional portfolio construction** at each rebalance: rank all eligible names, pick top decile longs and bottom decile shorts, size to target portfolio Greeks (beta, sector neutrality).
- **Capital allocation across scenarios:** if 8 scenarios fire on the same day for 8 different names, do not pretend each one independently gets full capital. Allocate proportionally to confidence and cap at the per-trade and per-portfolio limits.
- **Pair / spread trades on commodities:** model both legs simultaneously. P&L is `change_in_long − change_in_short`; margin is calculated on the net (CME SPAN cross-margins related contracts).
- **Borrow availability for the whole short book:** reject shorts where historical borrow was unavailable.
- **Correlation-aware sizing (optional, v3):** when adding a position, compute its correlation to the existing book; if correlation > 0.7 to a current position, halve the size.

These are *portfolio* concerns; a single-asset backtester won't surface them. Build the multi-asset engine from day one even if your first strategy is single-asset.

---

## 11. Common backtest fallacies and how this engine prevents each

| Fallacy | What it looks like | Engine prevention |
|---|---|---|
| **Look-ahead bias** | Using today's close to predict today's return | `asof_join` with `allow_exact_matches=False`; CI test for leakage |
| **Survivorship bias** | Backtesting on today's S&P 500 going back 10 years | `universe_membership` table; engine queries by date |
| **Future leakage in features** | Earnings surprise feature available before the print | `available_at` timestamps stamped at ingestion |
| **Free borrow** | Shorting GME in 2021 in a backtest | `borrow_availability` table; reject if not available |
| **Ignoring delistings** | Stock vanishes from universe at delisting; no loss recorded | Engine holds through delisting, marks loss |
| **Frictionless fills** | Buy at the close, sell at the close | Fill engine simulates next-bar limit + slippage; rejects unfillable orders |
| **Optimization on the test set** | Tuning parameters against the held-out period | Walk-forward + purged CV are mechanical; tuning happens only on train folds |
| **Selection bias** | Reporting Sharpe of best strategy from 50 tried | Deflated Sharpe; report `K` (strategies tried) in every tearsheet |
| **Regime overfit** | Trained 2010–2020, tested 2021, claims edge for all regimes | Combinatorial purged CV produces regime-spanning OOS distribution |
| **Backtest-live divergence** | "Worked in backtest, broken in live" | Same scenario manager + risk gateway code in both; parity test in CI |
| **Overstated Sharpe from autocorrelation** | Daily returns are autocorrelated; naive Sharpe over-claims | Use Newey-West standard errors when reporting Sharpe; report block-bootstrap CI |

---

## 12. CI integration

The backtester runs in CI on every PR, but in cheap mode:
- **Smoke backtest:** 1 year, 20 names, single strategy. Should run in < 60 seconds. Fail PR if Sharpe drops > 0.3 vs. the previous baseline (regression detection).
- **Unit tests:** `test_no_lookahead`, `test_replay_determinism`, `test_parity` always run.
- **Full backtests** (5+ years, 200 names) run nightly on a scheduled job; results posted to a Grafana dashboard.

Pin `numpy`, `pandas`, `lightgbm`, `numba` versions in `requirements-backtest.txt`. A library version drift can shift a borderline trade and propagate to a different P&L curve — not a real change in edge, just dependency noise.

---

## 13. Sequence of build (the order that minimizes wasted work)

1. **Data layer first:** asof timestamps, universe membership, corporate actions, futures roll. Without this, every later result is questionable.
2. **Frictions model second:** even a stub that just charges 5 bps on every trade. Lets you compare gross vs. net early.
3. **L1 vectorbt skeleton:** fast iteration on baseline strategy. Goal is to reproduce a known result (e.g., SPY momentum: positive Sharpe in the literature) so you trust the data.
4. **L2 event-driven engine** with the same data layer. Run the same baseline; verify L1 and L2 agree to within the noise of fill-model differences.
5. **Triple-barrier labeling + walk-forward harness:** before you train any production model, the labeling and CV story must be solid.
6. **L3 system runner:** plug in the scenario manager. Run a 2-year backtest with 1 scenario type. Verify the parity test passes.
7. **Add scenarios and models incrementally;** every new scenario gets its own backtest acceptance gate (hit rate, expectancy) before joining the live system.

Build in this order and you will not have to redo earlier work as later layers change.

---

## 14. Acceptance gates (what counts as "ready for paper trading")

A strategy is ready to graduate from backtest to paper trading only if **all** of:

- [ ] Walk-forward median Sharpe > 1.0 over 5+ years.
- [ ] Combinatorial purged CV 5th-percentile Sharpe > 0.5.
- [ ] Deflated Sharpe > 0.7 (after accounting for K strategies tried).
- [ ] Max drawdown ≥ 8% (if < 8%, probably overfit).
- [ ] Win rate ≤ 65% (if higher, probably overfit; *exception:* premium-selling strategies).
- [ ] No more than one losing year over the test period.
- [ ] Cost drag (gross − net) below 30% of gross return.
- [ ] Performance roughly stable across regimes (no single regime contributes > 60% of return).
- [ ] All CI tests green: no-lookahead, determinism, parity.

If any fails, do not promote. The cost of a wrong promotion is real money; the cost of one more iteration is hours.

---

*Document version 1.0. Companion to `IMPLEMENTATION_PLAN.md` and `final_requirements.txt`. Iterate as backtests reveal what the model and the market are actually doing.*
