# TraderWise — Validation Strategy & UI

> **Companion docs:** `IMPLEMENTATION_PLAN.md`, `BACKTESTING.md`, `LOCAL_DEV.md`, `final_requirements.txt`.
> **What this doc adds:** how you *prove* the system works at every layer, the UI that makes that proof visible, and the cadence at which you re-prove it as the system evolves.

The point of validation is not "we ran tests." It is: **before any layer is allowed to influence a real-money decision, there is a dashboard you can open and a number you can read that tells you the layer is producing what it claims to produce.** Without that, the system is a black box that loses money slowly while you debug models in isolation. With it, every layer carries its own confidence.

---

## 1. The validation pyramid

Five layers, each with its own gate. Each layer is gated by passing the layer below. Skip a layer and the gates above are meaningless.

```
                          ┌──────────────────────────────┐
                          │  L5  Live performance        │
                          │      (real money tracking)    │
                          └──────────────┬───────────────┘
                                         │
                          ┌──────────────────────────────┐
                          │  L4  Paper-trading parity    │
                          │      (paper P&L vs backtest) │
                          └──────────────┬───────────────┘
                                         │
                          ┌──────────────────────────────┐
                          │  L3  Strategy backtest       │
                          │      (Sharpe, MDD, gates)    │
                          └──────────────┬───────────────┘
                                         │
                          ┌──────────────────────────────┐
                          │  L2  Model & scenario        │
                          │      validation              │
                          │      (calibration, hit rate) │
                          └──────────────┬───────────────┘
                                         │
                          ┌──────────────────────────────┐
                          │  L1  Data & feature          │
                          │      integrity               │
                          │      (freshness, no leakage) │
                          └──────────────────────────────┘
```

Validation work happens **bottom-up during build, top-down during operation.** During build you cannot validate models on dirty data, so L1 first. During operation, the symptom you see is bad P&L (L5), and you debug downward to find which layer broke.

---

## 2. Layer-by-layer validation

### L1 — Data & feature integrity

**Question:** is the data correct, timely, point-in-time-safe?

| Check | How | Frequency | Pass criterion |
|---|---|---|---|
| **Freshness** | Latest bar timestamp per symbol vs. exchange-calendar last close | Every minute during market hours | Lag ≤ 15 min for slow path; ≤ 5 sec for fast path |
| **Completeness** | Expected vs. actual bar count per symbol per day | Daily after close | ≥ 99.5% of expected bars present; missing flagged |
| **No-look-ahead** | Per-feature `available_at < decision_time` test on a sampled grid of (symbol, t) | CI on every PR | Zero violations |
| **Survivorship-safe universe** | Backtest universe at any historical date is queried against `universe_membership`, not today's symbol list | CI test | Zero today-only symbols in historical universe |
| **Corporate-action alignment** | Adjusted-close continuity: `abs(adjusted[t] / unadjusted[t] - factor)` within tolerance across all split events | Nightly | Mismatches < 0.01% of events |
| **Schema drift** | Column types and ranges per table vs. baseline | Nightly | No silent type/range changes |
| **Feature drift (PSI)** | Population Stability Index per feature, current 21d vs. training distribution | Daily | PSI < 0.1 OK; 0.1–0.25 warning; ≥ 0.25 fail (retrain trigger) |
| **Duplicate / missing primary keys** | `COUNT(*) vs COUNT(DISTINCT pk)` | Hourly | Zero duplicates |

**Fail mode:** if L1 fails, **freeze model retraining and trading**. Fix data, then re-run all downstream validation. You cannot "trust the model and ignore the data warning" — every model output is a function of these inputs.

### L2 — Model & scenario validation

**Question:** are the models calibrated, stable, and not overfit? Are scenario triggers firing as expected?

#### Models

| Check | How | Frequency | Pass criterion |
|---|---|---|---|
| **Calibration** | Brier score on out-of-fold predictions; reliability diagram by decile | Per training cycle + rolling 60d in production | Slope of reliability line in [0.85, 1.15]; Brier < baseline |
| **Walk-forward Sharpe** | Per-quarter OOS Sharpe across all walk-forward folds | Per training cycle | Median > 1.0; 5th percentile > 0.5 |
| **Combinatorial purged CV** | Distribution of OOS Sharpe over N=50 train/test combinations | Per major model change | 5th percentile > 0.5 |
| **Deflated Sharpe** | Bailey & Lopez de Prado correction for K strategies tried | Before any promotion | Deflated Sharpe > 0.7 |
| **Feature importance stability** | Importance variance across CV folds vs. median | Per training cycle | Drop features with variance > 2× median |
| **Prediction drift** | PSI on prediction distribution, current vs. training | Daily | PSI < 0.25 |
| **Calibration drift** | 60-day rolling Brier vs. backtest baseline | Daily | Drift < 20% over 60 days |

#### Scenarios

| Check | How | Frequency | Pass criterion |
|---|---|---|---|
| **Birth rate per scenario type** | Births / trading-day, rolling 30d | Daily | Within ±50% of backtest birth rate |
| **Confirmation rate** | Confirmed / born, per scenario type | Daily | Within ±20% absolute of backtest |
| **Time to confirmation** | Median + p25/p75 minutes from birth → confirmed | Weekly | Within 2× of backtest median |
| **Invalidation pattern coverage** | Top-k invalidation reasons; do live patterns match backtest? | Weekly | Top-3 patterns match backtest top-5 |
| **Concurrent scenario count** | Max alive concurrently per session | Per session | ≤ 6 (alert at > 10; alert at flat 0 across full session) |
| **Hit-rate by type** | P&L of confirmed sessions, rolling 30d | Daily | Within ±20% of backtest hit rate per type |

**Fail mode:** if a scenario's hit rate diverges by > 20% from backtest for more than 30 days, **disable that scenario type** and run the post-mortem batch on its sessions to find the trigger weakness. Re-enable only after a hypothesis is verified offline.

### L3 — Strategy backtest

This is fully covered in `BACKTESTING.md` §14 (acceptance gates). The validation surface is the **tearsheet** plus a comparison view between strategy variants. Re-summarized for completeness:

A strategy must pass **all**:
- Walk-forward median Sharpe > 1.0 over 5+ years
- Combinatorial purged CV 5%ile Sharpe > 0.5
- Deflated Sharpe > 0.7
- Max drawdown ≥ 8% (lower → suspect overfit)
- Win rate ≤ 65% on directional (higher → suspect overfit)
- ≤ 1 losing year in test period
- Cost drag < 30% of gross return
- No regime contributes > 60% of return
- All CI tests green

If even one gate fails, the strategy does not graduate to paper trading.

### L4 — Paper-trading parity

**Question:** does the paper-trading P&L match the backtest distribution? If not, where is the gap?

This is where most retail systems silently die. The backtest says Sharpe 1.4; paper trading says Sharpe 0.3; the team starts blaming the broker, the data, the regime — when the actual gap is usually in fill modeling, latency, or a feature that was leaking in backtest.

| Check | How | Frequency | Pass criterion |
|---|---|---|---|
| **Trade-by-trade reconciliation** | For every paper trade, replay the same decision through the backtest engine using historical data up to decision time. Compare entry/exit prices and sizes. | Daily | ≥ 95% of paper trades reproduce within ±5 bps in backtest replay |
| **Distribution match** | KS test: paper trade returns vs. backtest trade returns over rolling 60d | Weekly | KS p-value > 0.05 |
| **Slippage realism** | Actual paper slippage vs. modeled slippage per asset class | Weekly | Within ±30% of model |
| **Latency profile** | Time from `scenario_confirmed` → order acknowledged | Per trade | < 5 sec at p95 |
| **Rejection coverage** | Risk gateway rejections per day, with reason codes | Daily | Distribution stable; no reason code spiking |
| **HITL skip / approval rates** | % approved vs. skipped vs. modified vs. ignored | Daily | Skip rate < 50% (above → notification fatigue or model quality drop) |

Run paper trading for **at least 4 weeks** before any live capital. The L4 gate to graduate to L5: **rolling 30-day paper Sharpe within ±0.4 of walk-forward median Sharpe.** If not within band, do not deploy capital — find the gap.

### L5 — Live performance

**Question:** is real-money P&L tracking expectation? When it doesn't, what failed?

| Check | How | Frequency | Pass criterion |
|---|---|---|---|
| **Rolling Sharpe vs. expectation** | 30-day rolling live Sharpe vs. backtest-implied 80% confidence interval | Daily | Inside CI |
| **Drawdown vs. circuit breakers** | Current drawdown vs. 8% / 10% halt thresholds | Continuous | Below; otherwise auto-halt |
| **Trade count vs. expectation** | 30-day live trade count vs. backtest expected | Weekly | Within ±30% |
| **Per-scenario P&L vs. backtest** | Cumulative P&L per scenario type vs. expected | Weekly | Within backtest 80% CI |
| **Cost drag delta** | Live cost drag vs. modeled | Monthly | Within ±20% |
| **System uptime during market hours** | Minutes lost per session | Continuous | < 1% downtime |
| **Audit trail completeness** | Every trade ties back to session_id → birth_state → model_version → code_version | Continuous | 100% |

**Fail-fast triggers** (per spec §14): single trade loss > 2× expected max → full halt; 90-day Sharpe < 0.5 → halt + post-mortem; calibration drift > 15% → freeze model and retrain.

---

## 3. The validation UI

### 3.1 Tech choice: Streamlit first, Next.js when needed

Per spec §12.5.6: build the validation UI as a **multi-page Streamlit app** for v1 and v2. Migrate select pages to Next.js only when (a) you want sub-second interactions, (b) you want mobile-quality layout, or (c) someone other than you starts using it.

Why Streamlit for now:
- Renders a working dashboard from ~50 lines of Python.
- Direct access to your existing pandas / SQLAlchemy / Qdrant clients — no API layer needed.
- Auto-refreshes on file save during development.
- Runs locally, binds to `localhost:8501`, costs nothing.

Why Next.js eventually:
- The Telegram bot covers the urgent mobile path; the dashboard's job is desk-bound deep inspection.
- Streamlit's reactive model becomes painful past ~10 interactive widgets per page.
- Real-time updates via WebSocket are cleaner in Next.js.

Folder layout:

```
app/dashboard/
├── streamlit_app.py             # entry point
├── pages/
│   ├── 1_data_health.py
│   ├── 2_feature_explorer.py
│   ├── 3_model_performance.py
│   ├── 4_backtest_viewer.py
│   ├── 5_scenario_inspector.py
│   ├── 6_session_replay.py
│   ├── 7_trade_journal.py
│   ├── 8_risk_monitor.py
│   ├── 9_paper_vs_backtest.py
│   └── 10_live_overview.py
├── components/                   # shared widgets (stat cards, sparklines)
├── queries/                      # SQL templates per page
└── theme/                        # dark theme + chart palettes
```

Each page is self-contained: opens its own DB connection, renders its own panels, refreshes independently.

### 3.2 Page-by-page spec

#### Page 1 — Data Health
**Purpose:** answer "is the data layer working?" in 5 seconds.
- Top strip: per-source freshness badges (Polygon, FRED, COT, Yahoo backup, news feed). Green if within tolerance, red otherwise.
- Bar count completeness heatmap (symbols × last 30 days).
- Last 7 days of feature drift PSI (top-10 most-drifted features highlighted).
- Schema drift log (new columns, type changes — empty most days).
- Survivorship-safety check: did any backtest in the last 7 days reference a symbol not in the historical universe at that date?

#### Page 2 — Feature Explorer
**Purpose:** sanity-check feature distributions and stability.
- Searchable feature list with summary stats (mean, std, p5/p50/p95).
- Per-feature: distribution histogram (training vs. last 21 days overlaid), autocorrelation, missingness over time.
- Cross-feature correlation heatmap (top 30 features) — spot redundancy.
- "Feature value timeline" mode: pick a symbol + date range + features → line chart with the value at every decision time. Used to manually verify no-look-ahead.
- Export to CSV for ad-hoc analysis.

#### Page 3 — Model Performance
**Purpose:** is each model calibrated, stable, and behaving like backtest predicts?
- Tabs per model (A, B′, C, D, meta).
- For each:
    - **Reliability diagram** with confidence bands (predicted P vs. observed P, by decile).
    - **Brier score** trend over last 90 days vs. backtest baseline.
    - **Feature importance** with stability stripes (variance across CV folds).
    - **Prediction distribution drift** PSI sparkline.
    - **Walk-forward Sharpe** per quarter, last 5 years.
    - **Decile lift chart** (top vs. bottom prediction decile P&L).
- "Compare versions" mode: select two model versions, see calibration deltas + decile lift deltas. Critical when promoting a retrain.

#### Page 4 — Backtest Viewer
**Purpose:** browse and compare backtest runs.
- Run picker: by date / strategy_name / git SHA.
- Tearsheet panel (the format from `BACKTESTING.md` §9): P&L, trade stats, attribution, validation, frictions.
- Equity curve with drawdown shading.
- Trade list (sortable, filterable by scenario type, sector, regime).
- "Compare runs" mode: pick 2–4 runs, overlay equity curves, diff stats side-by-side.
- Walk-forward distribution plot (Sharpe per quarter as box plot).
- Combinatorial purged CV violin plot (OOS Sharpe distribution).
- Export tearsheet to PDF.

#### Page 5 — Scenario Inspector
**Purpose:** understand what scenarios are firing now and historically.
- **Live tab:** all currently-active scenarios with state, age, key metrics, kill criteria status. Click → drill into the session.
- **Historical tab:** filter sessions by scenario_type / mode / outcome / date range. Aggregate stats: birth rate, confirmation rate, hit rate, avg P&L.
- **Per-type tab:** for each scenario type, plot rolling 30-day birth rate, confirmation rate, hit rate against backtest baseline as horizontal lines. The instant a live curve diverges from the baseline, you see it here.
- "Scenario diff" mode: select a date range with poor performance, compare birth_state distributions to a date range with good performance — find the regime that broke the scenario.

#### Page 6 — Session Replay
**Purpose:** the most important debugging tool. Pick any historical session and step through it candle by candle.
- Session search by `session_id` or filters.
- Timeline scrubber: drag to any candle in the session's life.
- At each step:
    - Price chart of the underlying with birth marker, confirmation marker, kill markers, exit marker.
    - Snapshot of every feature at that candle.
    - Snapshot of every model prediction at that candle.
    - Active scenario state and which conditions were met / not met.
    - If a trade was placed: order details, fill, current P&L.
- "Replay through the backtester" button: takes the session, runs it through the backtest engine, shows what would have happened. Used for L4 paper-vs-backtest reconciliation.
- "Re-run with this candidate change" button (advanced): apply a hypothetical model or scenario-rule change and replay; visualize counterfactual.

#### Page 7 — Trade Journal
**Purpose:** human-curated context layered on top of every trade.
- One row per trade: entry, exit, P&L, scenario_type, structure, your approval/skip/modify decision.
- Editable notes column — free text, your in-the-moment thinking.
- Tag system: `regime_change`, `news_driven`, `i_was_tired`, etc. Used later for post-mortem aggregation.
- Skip-reason analytics: % of skips by reason, P&L if you had taken them.
- Monthly retrospective generator: button that runs an LLM call over the month's journal + skip data and produces a structured retrospective.

#### Page 8 — Risk Monitor
**Purpose:** real-time view of risk vs. limits.
- Top strip: gross exposure, net exposure, net beta, total delta, sector concentrations, current drawdown — each as a gauge with limit lines.
- Open-position list with per-position dollar risk to stop, distance to stop in ATRs.
- Circuit-breaker status: is the kill switch armed? What would trip it next?
- Last 30 days of rule rejections by reason code.
- "Stress test" widget: apply a hypothetical −5% / −10% SPY shock and show portfolio P&L impact.

#### Page 9 — Paper vs. Backtest
**Purpose:** the L4 reconciliation page. The single most-watched page during weeks 11–16 of the roadmap.
- Side-by-side equity curves (paper rolling 30d vs. backtest equivalent on the same window).
- Per-trade reconciliation table: every paper trade replayed through the backtest engine; deltas in entry, exit, fill, P&L flagged.
- KS-test panel: paper return distribution vs. backtest return distribution; rolling p-value.
- Slippage realism: actual paper slippage vs. modeled, by asset class.
- "Gap explained" attribution: where did the difference between paper Sharpe and backtest Sharpe come from? (Slippage / latency / missed fills / regime mismatch.)

#### Page 10 — Live Overview
**Purpose:** the ops view once real money is deployed. Mostly hidden until you hit L5.
- KPI strip: today's P&L, MTD P&L, YTD P&L, current drawdown, system uptime.
- Open positions with live Greeks-equivalents (delta to SPY, dollar at risk).
- Today's signal log: time, scenario, action taken (auto / approved / skipped / modified), result.
- Upcoming macro events with blackout windows highlighted.
- 30-day live Sharpe vs. backtest CI (the chart).
- Recent kill-switch events log.
- Anthropic API spend last 7 days (cost discipline per spec §8.6).

### 3.3 Real-time vs. polled

- Pages 1, 5, 8, 10 want real-time (or near-real-time). Implement with Streamlit's `st_autorefresh(interval=15_000)` initially; migrate to Next.js + WebSocket when you outgrow it.
- Pages 2, 3, 4, 6, 7, 9 are inspection tools. Manual refresh is fine.

### 3.4 Auth and access

For local-only dev: no auth needed. Bind Streamlit to `localhost` and trust the laptop. When the dashboard moves to a remote VPS:
- Single Streamlit deployment behind Cloudflare Access (free for personal use), restricted to your email.
- Or basic-auth via `streamlit-authenticator` with a single account.

Do not expose the dashboard to the public internet without auth — it leaks model versions, positions, and signals.

---

## 4. Validation cadence

When does each thing get checked?

### Continuous (during market hours)
- Data freshness alarms.
- Live position risk gauges.
- Circuit-breaker checks before every order.
- WebSocket feed liveness.

### Per trade
- Risk gateway validation (already on the hot path).
- Kill criteria evaluation per candle for open positions.
- Audit-trail log entry.

### Daily (after close)
- Bar completeness check.
- Feature PSI per feature.
- Model calibration Brier (rolling 60d).
- Per-scenario hit-rate update.
- Daily P&L attribution.
- Post-mortem batch on dead sessions (LLM call, batched, cheap).
- Email/Telegram digest: today's signals, P&L, anomalies.

### Weekly
- Trade-by-trade paper-vs-backtest reconciliation.
- Skip-reason analytics review.
- Scenario time-to-confirmation distribution check.
- Schema drift report.
- Backup verification (TimescaleDB → S3 / Time Machine).

### Monthly
- Full tearsheet vs. backtest expectation comparison.
- Cost drag delta review.
- Notification-fatigue analysis (are you systematically ignoring the 11 PM signals?).
- Spend audit (Anthropic API + data feed costs).
- HITL retrospective: which signal types do you skip most? Which skip reasons predict actual failure?

### Quarterly
- Walk-forward retrain of all models.
- Universe-membership refresh (S&P 500 / Russell 1000 reconstitutions).
- Combinatorial purged CV re-run.
- Deflated Sharpe re-computation with updated K (strategies tried).
- Capital ramp decision per spec §13 v3.

### Per release / per PR
- Full CI: no-lookahead, replay determinism, parity test, smoke backtest.
- Schema migration dry-run on staging copy.
- Monitoring alert rule sanity check.

---

## 5. The phase-gate model (what to validate when)

Mapping validation to the `IMPLEMENTATION_PLAN.md` roadmap:

| Phase | Validation focus | Gate to next phase |
|---|---|---|
| **v0 — Foundations** | L1 data health: ingestion correctness, no-look-ahead, survivorship-safe universe. **Baseline strategy** (SPY mean-reversion or similar simple rule) backtests with Sharpe > 0.5 over 5 years. | Pass L1 + baseline gate. If baseline fails Sharpe 0.5, you have a data problem, not a model problem. |
| **v1 — Daily decision system** | L2: Models A/B′/C calibration, walk-forward Sharpe, feature importance stability. L3: full strategy backtest passes all 9 acceptance gates. Streamlit pages 1–4 + 8 live. | Pass all L3 acceptance gates from §2 / `BACKTESTING.md` §14. |
| **v2 — Intraday scenarios** | L2 scenarios: birth/confirmation/hit-rate matches backtest. Streamlit pages 5, 6, 9 added. ≥ 4 weeks of paper trading. | L4 gate: 30-day paper Sharpe within ±0.4 of walk-forward median Sharpe. |
| **v3 — Live capital ramp** | L5: rolling Sharpe inside backtest CI. Audit-trail completeness 100%. Streamlit page 10 live. | Capital ramp gates: $5K → $20K after 60 live days within expectation; $20K → $50K after 90 more. |
| **v4 — Selective auto-execute** | L5: per-scenario hit rate within ±5% of backtest over ≥ 50 live sessions. | Per-scenario auto-execute approval; never blanket. |

The gate at each phase is binary. If you do not pass, you do not proceed — you debug.

---

## 6. The reconciliation workflow (the most-used loop)

When something is off (paper Sharpe < expected; hit rate diverging; weird trade), the workflow is the same:

1. **Page 10 (Live Overview)** notices the anomaly.
2. **Page 9 (Paper vs Backtest)** quantifies the gap and attributes it (slippage / latency / signal quality).
3. **Page 5 (Scenario Inspector)** drills into the scenario type behind the anomalous trades.
4. **Page 6 (Session Replay)** opens the specific suspect session and walks through every candle.
5. **Page 3 (Model Performance)** checks whether the model whose prediction triggered the scenario is mis-calibrated.
6. **Page 2 (Feature Explorer)** inspects the inputs to that model on those dates.
7. **Page 1 (Data Health)** confirms the inputs were not corrupt or stale.
8. Form a hypothesis. Encode it as a fix or a test. Push. Re-run from §5's appropriate gate.

This is the loop that turns "the system is losing money" into "I know exactly which assumption broke." Without these pages, the same debugging takes days; with them, hours.

---

## 7. What "broken" looks like (failure-mode catalog)

Cross-reference for the validation UI. Each row maps a symptom to its likely cause and which page to start on.

| Symptom | Likely cause | Start on page |
|---|---|---|
| Live Sharpe < paper Sharpe by > 0.4 | Real-world frictions higher than modeled; or borrow / locate failures on shorts | 9, then 7 for individual trades |
| Paper Sharpe < backtest Sharpe by > 0.5 | Look-ahead bias in features; or unrealistic fill model | 9, then 2 to inspect feature timing |
| Win rate dropped > 10pp in 30 days | Regime change; or model calibration drift | 3 (calibration trend), then 2 (feature drift) |
| Scenario birth rate spiked 2× | Trigger logic too loose, or VIX regime change | 5 (per-type tab), then 2 |
| Scenario birth rate dropped to zero | Data feed broken, or scenario rule too tight | 1 (freshness), then 5 |
| Sharp drawdown not predicted by model | Model un-calibrated for current regime; or risk gateway not enforcing limits | 8, then 3 |
| You skip > 50% of signals | Notification fatigue, or model quality slipped | 7 (skip analytics), then 3 |
| Trades fail to fill repeatedly | Spread widened (microstructure regime), or limit too tight | 9 (slippage panel), then 5 |
| Risk gateway rejecting 30%+ of plans | Sizing model giving too-large outputs; or limits mis-tuned | 8, then 3 |
| LLM trade plans inconsistent / hallucinatory | Tool spec or RAG context stale; cache eviction | 6 (replay) + raw LLM logs |
| Anthropic spend spiking | LLM being called at wrong frequency (likely per-candle leak) | 10 (spend panel), then orchestrator logs |
| Feature PSI > 0.25 on a top-importance feature | Regime change or data-source change | 2, then 1 |

Build the failure-mode catalog into a help page in the dashboard so the next debugging cycle is mechanical, not improvisational.

---

## 8. Statistical tests to keep on hand (cheat sheet)

Implementations live in `app/core/ai/validation/`. Each is small (< 100 lines).

| Test | What it answers | When to use |
|---|---|---|
| **Population Stability Index (PSI)** | Has this distribution shifted? | Feature drift, prediction drift |
| **Kolmogorov–Smirnov** | Are these two samples from the same distribution? | Paper vs. backtest return reconciliation |
| **Brier score + reliability diagram** | Are predicted probabilities calibrated? | Model calibration |
| **Newey–West HAC standard errors** | What is the true Sharpe given autocorrelation? | Reporting Sharpe of any strategy |
| **Block bootstrap** | What is the confidence interval on Sharpe / MDD? | Backtest robustness |
| **Deflated Sharpe Ratio** | Did I overfit by trying many strategies? | Before any promotion |
| **Hansen's SPA test** | Is the best strategy actually better than chance given K trials? | Final selection between candidates |
| **Spearman rank correlation** | Are model rankings stable over time? | Cross-sectional ranker (Model D) drift |
| **DM (Diebold–Mariano)** | Is model A's forecast meaningfully better than model B's? | Comparing two models that look similar |

Plot all of these on the dashboard, not just compute them. A number on a Slack message is forgotten in 5 minutes; a chart on Page 3 is checked every day.

---

## 9. Day-zero validation skeleton (what to build first)

Even before you build all 10 pages, the minimum viable validation surface is:

1. **A `validation` package** (`app/core/ai/validation/`) with: `psi.py`, `calibration.py`, `ks.py`, `deflated_sharpe.py`, `purged_cv.py`. ~500 lines total.
2. **CI tests** for no-lookahead + replay determinism + parity. Three test files, ~200 lines.
3. **Streamlit Page 1 (Data Health)** + **Page 4 (Backtest Viewer)**. The two pages you'll open every day.
4. **A nightly job** that runs PSI + calibration + bar-completeness checks, writes results to a `validation_runs` table, and posts a Telegram digest.
5. **A `make validate` target** that runs the smoke backtest + all CI tests + writes a one-page report.

That's enough to validate the v0 → v1 transition. Add pages 2, 3, 5, 6, 7, 8, 9, 10 incrementally as the layers they validate come online.

---

## 10. The one paragraph version

Validation is structured as a five-layer pyramid (data → models/scenarios → strategy backtest → paper parity → live performance), each layer gated by the one below. Each gate is a number you can read on a Streamlit dashboard. The dashboard is a 10-page app: Data Health, Feature Explorer, Model Performance, Backtest Viewer, Scenario Inspector, Session Replay, Trade Journal, Risk Monitor, Paper-vs-Backtest, Live Overview. Build pages 1 + 4 first (data + backtests are what you need on day zero); add 5, 6, 9 when paper trading starts; add 8 + 10 when real money goes in. The single most-used page is Session Replay — pick any historical session and step through candle-by-candle to debug exactly what fired, what didn't, and why. Validation cadence is continuous (data + risk), daily (P&L + drift), weekly (paper reconciliation + skip analytics), monthly (full retrospective), quarterly (walk-forward retrain + capital ramp decision). The phase-gate model from `IMPLEMENTATION_PLAN.md` plus the acceptance gates from `BACKTESTING.md` §14 plus the L4 paper parity gate (paper Sharpe within ±0.4 of walk-forward Sharpe) are non-negotiable — fail any one and you do not advance. The goal is that before any layer influences a real-money decision, there is a chart on the dashboard that proves it works.

---

*Document version 1.0. Companion to `IMPLEMENTATION_PLAN.md`, `BACKTESTING.md`, `LOCAL_DEV.md`, and `final_requirements.txt`.*
