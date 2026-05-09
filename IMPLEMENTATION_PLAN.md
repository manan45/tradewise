# TraderWise — Implementation Plan
## Pivot to US Stocks + Commodities Swing Trading (No Options)

> **Source spec:** `final_requirements.txt` v1.6 (originally written for GLD options).
> **This document** re-targets that architecture to multi-asset US equity + futures swing trading,
> maps it against the current codebase, and lists every concrete change required.

---

## 0. The Pivot — What Changes vs. the Source Spec

The source spec is for **single-instrument options trading on GLD**. Roughly 40% of it transfers directly; the rest needs adaptation. Below is a clean table of what carries over, what changes, and what gets dropped.

| Concept in spec | Keep / Adapt / Drop | Reason |
|---|---|---|
| Predict *conditions* not prices | **Keep** — core principle is asset-agnostic | Holds for stocks/futures too |
| Vol regime classifier (Model A) | **Adapt** — replace GVZ with VIX + per-asset 21d realized vol rank | No GVZ for stocks |
| IV mispricing (Model B) | **Drop** — there's no IV without options | Replace with **Model B′: realized-vol expansion classifier** (will RV in next 10d exceed trailing 21d RV by ≥30%?) |
| Direction model (Model C) | **Keep** — multi-day return classifier | Now per-asset, cross-sectional |
| Scenario manager + sessions | **Keep** — most valuable architectural piece | Scenarios re-defined for cash equities/futures |
| Iron condors / calendars / spreads | **Drop** | Replace structures with **long/short shares**, **long/short futures**, **pair trades** |
| Greeks limits (delta/vega/theta) | **Drop** | Replace with **dollar-beta exposure**, **portfolio ATR risk**, **sector concentration** |
| GVZ + IV surface features | **Drop** | Replace with **VIX, VIX term structure (VIX/VIX3M), realized vol cones, cross-asset vol ranks** |
| 30–60 DTE selection | **Drop** | Replace with **planned hold horizon** (5–20 trading days for swings) |
| LLM orchestrator | **Keep** — same shape, different tool set | |
| Risk layer (rule-based, no LLM) | **Keep** | Re-express limits in cash terms, not Greeks |
| Backtest framework | **Keep & expand** — must support multi-asset | Replace `optopsy` with `vectorbt`/`zipline-reloaded`/custom |
| Telegram HITL UX | **Keep verbatim** | Cards just describe shares/contracts instead of strikes |
| Triple-barrier labeling, meta-labeling, Kelly sizing, deflated Sharpe | **Keep all** — these are the real edge levers | Universal techniques |
| Alpaca / IBKR execution | **Keep** | Alpaca for equities; IBKR for futures (commodities) |
| FOMC/CPI event blackouts | **Keep & expand** — add earnings blackouts per name | |

**The architectural skeleton (Data → Models → Sentiment → RAG → Scenario Manager → Orchestrator → Risk → Execution → HITL) stays identical.** What changes is the *content* in each layer.

---

## 1. Target System (Re-Specified for Stocks + Commodities Swing)

### 1.1 Universe
- **Equities:** S&P 500 + extended liquid Russell 1000 names. Optional: highly-liquid ETFs (SPY, QQQ, XLE, XLF, GLD, USO, IWM…). Initially cap at ~200 names.
- **Commodity futures (continuous front-month, rolled deterministically):**
    - Metals: GC (gold), SI (silver), HG (copper)
    - Energy: CL (WTI), NG (natgas)
    - Ags: ZC (corn), ZS (soy), ZW (wheat)
- **Hold horizon:** 3–20 trading days (swing). Daily decision cadence; 60-min intraday cadence inside the scenario manager.

### 1.2 Realistic targets (replaces spec §0)
| Metric | Realistic | Elite |
|---|---|---|
| Directional win rate | 50–55% | 55–58% |
| Reward-to-risk per trade | 1.6–2.2 : 1 | 2.2–3.0 : 1 |
| Sharpe (after costs) | 1.0–1.5 | 1.5–2.2 |
| Max drawdown | 12–18% | 8–12% |
| Annualized return on capital deployed | 15–25% | 25–40% |
| Avg holding period | 5–12 days | — |
| Turnover (annualized) | 4–8× | — |

If a backtest shows >65% win rate, Sharpe >3, or MDD <8% over 5+ years on a directional swing system — **assume overfitting**.

### 1.3 Forecasting models (revised)
| Model | Target | Algo | Use |
|---|---|---|---|
| **A — Vol regime** | VIX in top tercile in 5 trading days; per-asset 21d RV rank in top tercile | LightGBM multiclass | Reduce sizing in high-vol regimes; widen stops |
| **B′ — RV expansion** | Will next-10d RV exceed trailing-21d RV by ≥30%? | LightGBM binary, isotonic-calibrated | Avoid new entries before vol expansion; tighten stops on existing |
| **C — Direction (per asset)** | Will return over next 5 trading days exceed +1 ATR? (and symmetric short target) | LightGBM, regime features included | Long/short signal; combined with rank |
| **D — Cross-sectional rank** *(new)* | Rank all eligible names by 5-day expected return decile | LightGBM ranker (LambdaRank) | Picks the best 10 longs / 10 shorts each day |
| **E — Pair / relative-value** *(commodities)* | Will GC/SI ratio mean-revert to 60d median within 10 days? Same for GC/HG, CL spread, etc. | LightGBM binary | Spread trades on commodities (built but disabled until v3) |

### 1.4 Feature universe (40–60 features per asset)
Same families as spec §4, but re-cast:
- **Trend/momentum:** returns 5/10/21/63d, distance from 20/50/200 SMA, ADX, MACD histogram.
- **Mean reversion:** z-score vs 20-day MA, RSI(14) as feature, Bollinger %B.
- **Vol features:** 21d realized vol + rank, vol-of-vol, ATR(14) % of price, VIX level + rank, **VIX/VIX3M term-structure ratio** (replaces GVZ term structure), Garman–Klass / Yang–Zhang RV estimators.
- **Macro (carries over verbatim):** real yield (DFII10), DXY, breakeven (T10YIE), 2s10s, M2 growth.
- **Cross-sectional:** sector-relative momentum, beta-adjusted residual returns, **liquidity rank** (ADV / spread).
- **Cross-asset:** Gold/silver, gold/copper, copper/oil, equity/credit (HYG vs SPY).
- **Fundamental (equities only, slow features):** earnings surprise z-score, revisions breadth, valuation z-score (P/E vs sector median). Survivorship-bias-safe loaders are **mandatory**.
- **Microstructure (Tier 1 of spec §4.7):** quoted spread % of mid, top-of-book size, 5-min spread series.
- **Event:** days to next earnings (per name), days to next FOMC/CPI/NFP, imminence flag (≤48h).

All feature transforms use **rank/percentile**, not raw values (spec rule, kept).

### 1.5 Scenario types (re-defined for cash markets)
| Scenario | Birth trigger | Confirmation | Invalidation | TTL |
|---|---|---|---|---|
| **Trend continuation** | Price > 50d SMA, ADX > 20, 5d return in top quintile | Pullback to 10d EMA holds + new 20d high on volume ≥ 1.3× avg | Close back below 20d SMA | 10 trading days |
| **Mean reversion (oversold)** | RSI(14) < 25 AND z-score < −2 AND name in liquid universe | Daily close above prior bar high + RV stops expanding | New 20d low on close | 5 trading days |
| **Breakout** | Close above 50-bar Donchian channel + volume ≥ 1.5× 20d avg | Retest holds within 2 bars; no immediate fade | Close back inside the channel | 7 trading days |
| **Vol expansion (risk-off)** | VIX +15% in 2 days or VIX/VIX3M crosses 1.0 | VIX closes in top quartile; SPY 20d RV expands ≥ 30% | VIX retraces 50% within 3 days | 5 trading days |
| **Vol crush (risk-on)** | VIX in 90th+ percentile then 3 consecutive lower closes | VIX/VIX3M back below 0.95 | New VIX high | 5 trading days |
| **Earnings drift / PEAD** *(equities)* | Earnings surprise > 2σ + opening gap ≥ 3% in surprise direction | First-day close holds the gap; volume ≥ 2× ADV | Gap fill within 3 days | 15 trading days |
| **Commodity term-structure flip** *(futures)* | Front-month vs 3-month basis crosses zero | Holds for 3 sessions + COT positioning supportive | Reverts within 2 sessions | 10 trading days |

These are deterministic state machines. **LLM is invoked only at `confirmed` transitions.**

### 1.6 Risk layer (cash-equivalent, replaces spec §10)
Hard, code-enforced:
- **Per-trade max loss:** 1.0% of equity (stop = entry ± k·ATR; size = max-loss / per-share risk).
- **Concurrent open positions:** ≤ 8 single names + ≤ 3 commodity futures.
- **Per-sector concentration:** ≤ 25% of gross exposure.
- **Net beta to SPY:** −0.4 to +0.6 (not market-neutral, but bounded).
- **Gross exposure:** ≤ 150% (1.5× equity); start at 100%.
- **Per-name max position:** 8% of equity.
- **Drawdown circuit breaker:** 30-day rolling P&L < −8% → halt for 5 sessions, mandatory review.
- **Earnings blackout:** no new entries within 2 trading days of an earnings date for the name (use cached earnings calendar).
- **Macro blackout:** no new entries 24h before FOMC/CPI/NFP except event-specific scenarios.
- **Sizing formula:** `f = 0.25 × Kelly` using calibrated probability from Models C/D, capped at 1.0% per trade.
- **Intraday rate limit:** ≤ 6 `scenario_confirmed` per session.

### 1.7 Execution
- **Equities → Alpaca** (paper first, live later). Limit orders only; mid-or-better with 30s escalation rule (spec §15.9).
- **Futures → IBKR** (paper first). Contract roll calendar built into the data layer.
- **Auto-close (kill criteria) MUST run server-side** (broker-side stop orders + a watchdog process). UAE timezone means user is asleep when many stops fire.

### 1.8 Tech stack (revised)
| Layer | Choice | Notes |
|---|---|---|
| Time-series | **TimescaleDB** | Migrate from plain Postgres |
| Ephemeral state | **Redis** with TTL | New |
| Vector store | **Qdrant** (or pgvector to start) | New |
| Tabular models | **LightGBM** primary, CatBoost ensemble | Replace TF/Keras for tabular; keep PatchTST as optional sequence model |
| Sentiment | **FinBERT**, optional Qwen2.5-3B + LoRA | Already in deps, never wired |
| Embeddings | **BGE-large-en-v1.5** | New |
| Re-ranker | bge-reranker-base | New |
| Orchestrator LLM | Claude Sonnet 4.6 (Tier 1) → Haiku 4.5 (Tier 2) → FinBERT (Tier 3) | Per spec §8.6 |
| Serving | Triton on K8s (or local FastAPI inference until needed) | Defer Triton until prod |
| Bus | Replace Kafka with **RabbitMQ** OR keep Kafka but add RabbitMQ semantics via topics | RabbitMQ is simpler at this scale; Kafka is fine if already deployed. Pick one. |
| Realtime | Alpaca/IBKR WebSocket; FRED/yfinance polling for slow path | New |
| Backtest | `vectorbt` for fast iteration + custom event-driven engine | New |

---

## 2. Current State — Anchor Inventory

(Verified by codebase walk against `app/`, `infra/`, `requirements.txt`. Cite paths so you can re-verify.)

### 2.1 What exists that helps
- FastAPI shell (`app/api/main.py`) with suggestions endpoint, session lookup, prediction lookup.
- Yahoo Finance connector stub (`app/connectors/yahoo_finance.py`) — minimal but right idea for US equities.
- Postgres + SQLAlchemy + Alembic plumbing (`app/connectors/postgres_client.py`, `migrations/`).
- Domain entities for `Stock`, `Commodity`, `Index`, `MutualFund` already separated (`app/core/domain/entities/`).
- Repository interfaces present (`app/core/domain/interfaces/*`) — clean ports.
- An AI sub-system already split into the right folders: `analysis/`, `core_ml/`, `features/`, `learning/`, `portfolio/`, `session/`, `feedback/`, `reinforcement/`. The shape matches the spec; the *contents* don't.
- A risk-manager skeleton (`app/core/ai/portfolio/risk_manager.py`) with VaR + leverage limits.
- Session manager skeleton (`app/core/ai/session/session_manager.py`, `trading_session.py`).
- Docker Compose with Postgres + Kafka + Zookeeper (`infra/docker-compose.yml`).
- Trade-suggestion use case (`app/core/use_cases/...`) — entry point we can re-wire.

### 2.2 What exists but conflicts with the pivot
- **DhanHQ connector** (`app/connectors/dhanhq/`): Indian broker — keep as a reference adapter but it is not on the live path. Mark as legacy.
- **CLI ingestion to Indian commodities** (`app/main.py` → MCX): rewire to US ingestion.
- **Option-data entities** (`app/core/domain/entities/option_data.py`, `OptionChain`, `OpenInterest`): unused on the pivot path. Either delete or quarantine in `legacy/`.
- **TF/Keras + Stable-Baselines3 RL stack** (`requirements.txt`, `app/core/ai/learning/learning_system.py`, `app/core/ai/reinforcement/*`): swing trading on tabular features is a LightGBM problem, not an RL problem. RL is sample-inefficient and brittle (spec §15.11). Keep the folder for now, but it does not produce production signals.
- **Indian market hours + currency assumptions** scattered through services: audit and replace with US/Eastern + UAE-display.

### 2.3 Hard gaps vs. target
1. **No LLM orchestrator** — no Claude/Anthropic SDK integration, no tool definitions, no structured-output validation, no prompt caching.
2. **No scenario manager** — only naming exists, no state machines, no candle-driven update loop, no event emission.
3. **No sessions table / case-based retrieval** — the highest-leverage idea in the spec (§8.7) is absent.
4. **No backtester** — cannot validate any strategy on history.
5. **No paper-trading loop** — Yahoo/Dhan cannot fill orders.
6. **No US broker integration** — no Alpaca, no IBKR.
7. **No news/sentiment pipeline** — `transformers` is in deps but unused; no FinBERT, no labeled corpus.
8. **No RAG knowledge base** — no Qdrant, no embedder, no chunker.
9. **No TimescaleDB / Redis** — wrong stores for the shape of data we need.
10. **No risk-layer enforcement gateway** — risk module computes warnings; nothing rejects an order.
11. **No execution hooks for kill criteria / OCO orders.**
12. **No Telegram bot, no HITL dashboard.**
13. **No walk-forward / purged-CV / triple-barrier labeling** — spec §15 levers all absent.
14. **Survivorship bias** — current data ingestion has no concept of point-in-time membership for an equity universe. This is a silent killer of any equity research backtest.

---

## 3. Gap-by-Gap Change List

Each row is a single concrete change with its *file/folder*, an *action*, and the *reason* it matters. Implement top-down inside each phase.

### 3.1 Data layer
| # | Change | Where | Why |
|---|---|---|---|
| D1 | Replace plain Postgres with **TimescaleDB** image | `infra/docker-compose.yml` | All time-series queries (OHLC, candles, sessions) get hypertable acceleration. |
| D2 | Add **Redis** service | `infra/docker-compose.yml` + new `app/connectors/redis_client.py` | Ephemeral scenario state with TTL keys. |
| D3 | Add **Qdrant** service (or use pgvector extension) | `infra/docker-compose.yml` | RAG + case-based retrieval over `birth_embedding`. |
| D4 | Build US equity ingestion: **Polygon** (preferred) or **Alpaca Market Data** for daily + 1-min bars | New `app/connectors/polygon_client.py` | Yahoo is fine for prototyping; not for production. |
| D5 | Build **commodity futures ingestion** with deterministic continuous-contract roll calendar | New `app/connectors/databento_client.py` (or Barchart) | Continuous front-month is non-trivial; document the roll rule once and reuse. |
| D6 | Build **macro ingestion** (FRED: DFII10, T10YIE, DGS10, DGS2, DTWEXBGS, M2SL) | New `app/pipelines/ingest_macro.py` | Carries over from spec §3.2 verbatim. |
| D7 | Build **VIX + VIX3M** ingestion + term-structure feature | `app/pipelines/ingest_vix.py` | Replaces GVZ. |
| D8 | Build **CFTC COT** weekly ingestion for tracked commodities | `app/pipelines/ingest_cot.py` | Carries over from spec §3.4. |
| D9 | Build **point-in-time universe membership table** (S&P 500 historical constituents) | New table `universe_membership` | Eliminates survivorship bias. **Non-negotiable.** |
| D10 | Build **earnings calendar** ingestion (Polygon, Finnhub, or Alpaca news API) | `app/pipelines/ingest_earnings.py` | Drives per-name event blackouts. |
| D11 | Add a **two-speed real-time consumer**: slow path = poll every 1–5 min and aggregate to 15-min bars; fast path = WebSocket only at order placement | `app/services/realtime/` | Spec §3.1 — keeps cost low while preserving fill quality. |
| D12 | Mark `dhanhq/` as legacy | Move to `app/connectors/legacy/dhanhq/`, remove from imports | Pivot away from Indian markets. |

### 3.2 Feature engineering
| # | Change | Where | Why |
|---|---|---|---|
| F1 | Re-implement `feature_generation.py` to a **registry pattern**: each feature is a `(name, fn, dependencies, asset_class)` record | `app/core/ai/features/feature_generation.py` | Currently monolithic; needs to scale to ~50 features. |
| F2 | Add **rank/percentile transforms** on all numerical features (rolling 252-day window) | New `app/core/ai/features/transforms.py` | Spec §4: rank features are far more robust across regimes. |
| F3 | Add **realized vol estimators** (Garman–Klass, Yang–Zhang, Parkinson) | `app/core/ai/features/vol_features.py` | Better than close-to-close. |
| F4 | Add **VIX term-structure feature** (`vix / vix3m`) | `app/core/ai/features/vol_features.py` | Replaces GVZ term flips from spec §3.3. |
| F5 | Add **cross-sectional sector-relative momentum** (residual after sector beta-adjustment) | `app/core/ai/features/cross_sectional.py` | Required for Model D ranker. |
| F6 | Add **microstructure Tier 1**: quoted spread, top-of-book size, 5-min spread series | `app/core/ai/features/microstructure.py` | Spec §4.7 Tier 1 — execution gate + scenario birth gate. |
| F7 | **Strict no-look-ahead audit**: every feature stamped with `available_at` timestamp | Add helper `app/core/ai/features/asof.py` | Standard cause of fake backtest alpha. |

### 3.3 Models (replace current TF/RL stack)
| # | Change | Where | Why |
|---|---|---|---|
| M1 | Add `lightgbm`, `mlfinlab`, `scikit-learn>=1.4` to deps | `requirements.txt` | Tabular finance gold standard. |
| M2 | Implement **Model A** (vol regime) | `app/core/ai/models/vol_regime.py` | New. |
| M3 | Implement **Model B′** (RV expansion classifier) | `app/core/ai/models/rv_expansion.py` | Replaces spec's IV mispricing model. |
| M4 | Implement **Model C** (per-asset directional) | `app/core/ai/models/direction.py` | Calibrated probability. |
| M5 | Implement **Model D** (cross-sectional LambdaRank) | `app/core/ai/models/cross_section.py` | New for multi-asset. |
| M6 | Implement **Model E** (commodity pair mean reversion) — optional v3 | `app/core/ai/models/pairs.py` | Optional. |
| M7 | Implement **walk-forward training harness** + **purged k-fold CV** with 5-day embargo | `app/core/ai/training/walkforward.py` | Spec §5 + §15.4. |
| M8 | Implement **isotonic calibration** on validation folds + diagonal-curve check | `app/core/ai/training/calibration.py` | Calibrated probabilities are required for Kelly sizing. |
| M9 | Implement **triple-barrier labeling** | `app/core/ai/training/labels.py` | Spec §15.1 — single biggest precision lever. |
| M10 | Implement **meta-labeler** (second-stage filter) | `app/core/ai/models/meta_label.py` | Spec §15.2. |
| M11 | Quarantine current TF/Keras `learning_system.py` as `legacy/learning/` | `app/core/ai/legacy/` | RL is wrong tool for swing tabular finance; remove from prod path. |

### 3.4 Sentiment + RAG
| # | Change | Where | Why |
|---|---|---|---|
| S1 | Stand up **FinBERT** sentiment service (Triton or local FastAPI) | `app/services/sentiment/` | Spec §6. |
| S2 | Build **labeled corpus loader** (FOMC archives, Reuters/Polygon news) | `app/pipelines/ingest_news.py` | Required to fine-tune. |
| S3 | LoRA fine-tune FinBERT on hand-labeled sample (~2k docs) | One-off notebook in `experiments/sentiment_finetune/` | Spec §6.2. |
| S4 | Build **RAG ingestion** for Natenberg/Sinclair/Hull/Lopez-de-Prado + your trade journal | `app/pipelines/ingest_rag.py` | Spec §7. |
| S5 | **Hybrid retrieval** (BM25 + dense, RRF-fused) + bge-reranker | `app/services/rag/` | Spec §7.3. |

### 3.5 Scenario manager + sessions
| # | Change | Where | Why |
|---|---|---|---|
| SC1 | Implement `Scenario` base class per spec Appendix D | `app/core/ai/scenarios/base.py` | The skeleton. |
| SC2 | Implement 3 scenario subclasses first (trend continuation, mean reversion, vol expansion) | `app/core/ai/scenarios/` | Spec §13 v2 — start small. |
| SC3 | Implement `ScenarioManager.on_candle()` driven by 15-min bars (60-min for swing-heavy assets) | `app/core/ai/scenarios/manager.py` | Numerical updates only; no LLM in this loop. |
| SC4 | Create **`sessions` hypertable** with the schema in spec §8.7.2 | `migrations/` new revision | The system's long-term memory. |
| SC5 | Implement **birth_state snapshot** writer (single source of truth) | `app/core/ai/scenarios/persistence.py` | Critical: cannot retrofit features later without invalidating embeddings. |
| SC6 | Compute **birth_embedding** at session birth (concat normalized features → optional encoder) | `app/core/ai/scenarios/embedding.py` | Powers case-based retrieval. |
| SC7 | Implement **case-based retrieval** (top-k cosine over `sessions.birth_embedding` filtered by scenario_type, mode∈{live,paper}) | `app/services/retrieval/case_based.py` | Spec §8.7.5 — the LLM reasons from your *actual* track record. |
| SC8 | Implement **scenario events on RabbitMQ** (or Kafka topic) — only on lifecycle transitions | `app/services/messaging/` | Hard rate limit: ≤ 6 `scenario_confirmed` per session. |
| SC9 | Implement **daily post-mortem batch** that LLM-summarizes resolved sessions | `app/jobs/post_mortem.py` | Spec §8.5.6. |

### 3.6 LLM orchestrator
| # | Change | Where | Why |
|---|---|---|---|
| O1 | Add `anthropic` SDK + **prompt caching** | `requirements.txt`, `app/services/llm/anthropic_client.py` | Spec §8 + cost discipline. |
| O2 | Define **tools** per spec §8.3 (get_market_state, get_model_predictions, get_sentiment, search_knowledge_base, get_active_scenarios, get_current_positions, get_upcoming_events, compute_position_size, propose_trade) | `app/services/llm/tools.py` | Tool use is how the LLM gets structured inputs. |
| O3 | Implement **decision loop**: triggered → gather → 1–3 candidates → propose_trade → return JSON | `app/services/llm/orchestrator.py` | Spec §8.4. |
| O4 | Enforce **structured output** with pydantic schema validation | `app/services/llm/schemas.py` | Spec §8.5 — every trade plan is a typed object. |
| O5 | Implement **tiered routing**: Sonnet 4.6 (T1), Haiku 4.5 (T2), FinBERT (T3) | `app/services/llm/routing.py` | Spec §8.6. |
| O6 | Add **trigger sources**: daily cron at open/close + RabbitMQ subscriber on `scenarios.confirmed` + macro-event scheduler | `app/jobs/orchestrator_triggers.py` | Spec §8.2. |
| O7 | **Never** call LLM per candle, per tick, for support/resistance, or in backtests. Add a guard layer. | `app/services/llm/guards.py` | Spec §8.6.4 — explicit cost trap. |

### 3.7 Risk + execution
| # | Change | Where | Why |
|---|---|---|---|
| R1 | Promote `risk_manager.py` from advisory → **gateway** that validates and can reject orders | `app/core/ai/portfolio/risk_manager.py` | Spec §10 — no LLM override. |
| R2 | Re-express limits in **cash terms** (per-trade max loss, sector concentration, gross exposure, net beta to SPY) | Same | Replaces Greeks limits. |
| R3 | Implement **Kelly sizing with calibrated probability**, capped at 1% per trade | `app/core/ai/portfolio/sizing.py` | Spec §15.3 — highest leverage-per-line lever. |
| R4 | Implement **drawdown circuit breaker** that flips a Redis `system_state.trading_halted` flag | `app/services/risk/circuit_breaker.py` | Spec §12.5.8 — every component checks before acting. |
| R5 | Implement **Alpaca execution adapter** | `app/connectors/alpaca_client.py` | US equities. |
| R6 | Implement **IBKR execution adapter** | `app/connectors/ibkr_client.py` | US futures. |
| R7 | Implement **paper-trading mode** via Alpaca/IBKR sandboxes (preferred over a custom simulator for live realism) | Same adapters with `mode='paper'` flag | Less code than rolling our own. |
| R8 | Implement **broker-side OCO / stop orders** so kill criteria fire while user is asleep | Inside execution adapters | UAE timezone reality. |
| R9 | Implement **earnings + macro blackout checker** consulted on every new entry | `app/services/risk/blackouts.py` | Per-name and global. |

### 3.8 Backtesting
| # | Change | Where | Why |
|---|---|---|---|
| B1 | Add `vectorbt` + custom event-driven engine | New `app/backtest/` | Spec §11. |
| B2 | **Backtest replays through the same `ScenarioManager.on_candle()` as live**, only `mode` differs | `app/backtest/engine.py` | Spec §8.7.4 — research/production parity. The single-most-important architectural rule. |
| B3 | Implement **realistic frictions**: limit-order fill model, mid+1c slippage, $0.005/share commission for equities, $1.20/contract for futures | `app/backtest/frictions.py` | Spec §11. |
| B4 | Implement **walk-forward driver** that retrains quarterly | `app/backtest/walkforward_driver.py` | Spec §5. |
| B5 | Implement **combinatorial purged CV** + **deflated Sharpe** reports | `app/backtest/stats.py` | Spec §15.4 + §15.5. |
| B6 | Run survivorship-bias-safe equity backtests using `universe_membership` table | Wire to D9 | Without this, every backtest lies. |

### 3.9 HITL (Telegram + dashboard)
| # | Change | Where | Why |
|---|---|---|---|
| H1 | Build **Telegram bot** with inline approve/skip/modify keyboards | `app/services/telegram/` | Spec §12.5.3. |
| H2 | Notification card format per spec §12.5.4 | Same | 5-second readable / 10-second actionable. |
| H3 | **Skip-reason capture** → meta-labeler training data | `app/services/telegram/skip_handler.py` | Spec §12.5.5 — your skips are gold. |
| H4 | **Streamlit dashboard** with 5 panels (active sessions, today's signals, open positions, portfolio risk, upcoming events) | `app/dashboard/` | Spec §12.5.6. |
| H5 | **Kill switch** as Redis flag honored by every component | Wire to R4 | Spec §12.5.8. |
| H6 | **SMS fallback** via Twilio for high-confidence signals missed at 11pm UAE | `app/services/twilio/` | Spec §12.5.7. |

### 3.10 Operations + monitoring
| # | Change | Where | Why |
|---|---|---|---|
| Op1 | Add **Prometheus + Grafana** to compose | `infra/docker-compose.yml` | Spec §12. |
| Op2 | Emit metrics: feature drift (PSI), calibration drift (rolling Brier), P&L vs expectation, freshness, active scenario count, scenario hit-rate by type | `app/services/monitoring/` | Spec §12 + §14. |
| Op3 | Implement **failure-mode triggers** per spec §14 as alert rules | `infra/grafana/alerts/` | Auto-halt on hits. |
| Op4 | Standardize on **one message bus**: RabbitMQ or Kafka, not both | Pick + delete the other | Don't carry both. |
| Op5 | **Audit trail**: every notification, approval, order, fill, kill-switch event tied to `session_id` in TimescaleDB | `app/services/audit/` | Spec §12.5.9. |

---

## 4. Phased Roadmap (Re-Calibrated for the Pivot)

The spec's 16-week roadmap stays roughly intact. Reordering for the multi-asset, no-options reality:

### v0 — Foundations (weeks 1–2)
- D1, D2, D3 (TimescaleDB, Redis, Qdrant in compose).
- D9 (point-in-time universe table) — **do this before any modeling**.
- D4, D5, D6, D7, D8, D10 (US equity, commodity, macro, VIX, COT, earnings ingestion).
- B1–B3 (skeleton backtester + frictions).
- **Gate to v1:** baseline strategy ("buy SPY when 5-day return < −5% and VIX > 25; exit at +3% or 5d") backtests with Sharpe > 0.5 over 5 years using the new pipeline. If not, debug data before any ML.

### v1 — Daily decision system (weeks 3–10)
- F1–F7 (feature engineering).
- M1–M4, M7–M9 (Models A, B′, C, walk-forward, calibration, triple-barrier).
- S1–S3 (FinBERT live).
- S4–S5 (RAG knowledge base).
- O1–O7 (LLM orchestrator, daily-cron-only initially).
- R1–R4, R9 (risk gateway, sizing, circuit breaker, blackouts).
- R5 (Alpaca paper trading).
- H1–H4 (Telegram bot + Streamlit dashboard).
- **Gate to v2:** ≥ 4 weeks of paper trading; live P&L within ±20% of backtest expectation.

### v2 — Intraday scenario manager (weeks 11–14)
- D11 (real-time two-speed consumer).
- SC1–SC9 (scenarios + sessions + case-based retrieval).
- M5 (cross-sectional ranker).
- Add scenarios incrementally: trend continuation → mean reversion → vol expansion → breakout → vol crush → PEAD.
- O6 with RabbitMQ subscription on `scenarios.confirmed`.
- R6 (IBKR for futures).
- M10 (meta-labeler) trained on first 30 days of paper sessions + skip-reason data.
- **Gate to v3:** scenario hit rates within 20% of backtest hit rates.

### v3 — Production hardening (weeks 15+)
- M11 (kill the legacy TF/RL stack).
- B4–B6 (walk-forward driver, deflated Sharpe, survivorship-safe full backtests).
- Op1–Op5 (full monitoring + audit).
- M6 (commodity pair model — optional).
- Capital ramp: $5K → $20K after 60 live days within expectation; $20K → $50K after 90 more.
- Selective auto-execution **only** for scenario types with ≥50 live sessions and hit rate within 5% of backtest.

---

## 5. Enhancements & Optimizations (How to Make This "Perfect")

Beyond the spec, here are the changes that move this from "good systematic system" to "compounding edge":

### 5.1 Multi-asset alpha (this is the big one for stocks)
- **Cross-sectional ranker (Model D)** is the single biggest expected-return lever for an equity universe — it picks the best 10 longs and 10 shorts from 500 candidates instead of evaluating each name in isolation. Time-series-only models leave most of the equity edge on the table.
- **Sector-neutral construction**: build the long book and short book to roughly cancel sector exposure, then take residual factor risk explicitly.
- **Beta-hedge with SPY** to bound net market exposure inside the spec's beta band.

### 5.2 Survivorship-safe everything
- The point-in-time universe table is mentioned in §3.1 D9, but it deserves its own callout. Without it, every equity backtest is wrong by ~3–5% annualized. Run every model + backtest with the historical S&P 500 / Russell 1000 membership as of the rebalance date.

### 5.3 Real-time vs. batch separation
- Spec §3.1 introduces a two-speed architecture (slow path for analysis, WebSocket only at order placement). Keep it. The cost difference between polling and a 24/7 WebSocket on 200 names is enormous and the predictive content of intra-second data at swing horizon is zero.

### 5.4 Hard separation: compute vs. reason
- Per spec §8.6.4, the LLM **never** identifies S/R, never recognizes patterns, never trends-spots. All of that is deterministic Python (peak detection + DBSCAN, ADX, Mann–Kendall). The LLM only reasons over structured outputs of those computations. Build the guard layer (O7) and enforce it in code review.

### 5.5 Sessions table is the moat
- Spec §8.7 calls this "the single highest-leverage architectural decision." Prioritize SC4–SC7 above almost everything else once the data layer is up. Once sessions accumulate, case-based retrieval gives the LLM empirical grounding from your actual track record, not generic principles. This is what compounds.

### 5.6 Deflated Sharpe + combinatorial purged CV from day one
- Most retail builds get into trouble because they pick the best of N strategies from a single backtest. Deflated Sharpe corrects for selection bias; combinatorial purged CV gives a *distribution* of out-of-sample Sharpe, not a point estimate. Cheap to add, prevents months of chasing fake alpha.

### 5.7 Triple-barrier labeling + Kelly sizing
- Spec §15.1 + §15.3. The first changes what the model learns; the second changes how much capital each trade gets. Together, in our experience, these are the difference between a Sharpe 0.6 system and a Sharpe 1.2 system. They're also the cheapest to implement.

### 5.8 Meta-labeler trained on YOUR skips
- Spec §15.2 + §12.5.5. Once HITL is live, every Skip becomes a labeled negative. After ~200 skip events, train a meta-model on `(primary_signal_features, skip_reason → outcome)`. It will start filtering low-quality primary signals upstream — typically +5–15% absolute win rate at a 30% trade-frequency cost. Net Sharpe-positive almost always.

### 5.9 Operational alpha (underrated)
- Spec §15.9: limit-at-mid → escalate to mid+1c after 30s; avoid first 15 minutes; cancel-and-replace, never market in/out; trade the most-liquid expiries (or in our case, the most-liquid names). This routinely adds 20–40 bps per round-trip across a year.

### 5.10 Fundamental + alt-data layer (after baseline is solid)
- Earnings surprise, revisions breadth, valuation z-score per sector, short interest changes. Slow features, monthly refresh cadence. Add only after baseline ML is profitable; otherwise you're optimizing into noise.

### 5.11 Don't add (cost traps from spec §15.11)
- More LLM layers, social media sentiment, deep learning on tabular, RL, crypto/equity correlations as predictors, higher-frequency data on swing strategies. Each of these has cost a quant team months of wasted effort.

### 5.12 What "perfect" actually means here
- A *positive-expectancy systematic process* with Sharpe 1.2–1.5, MDD 12–18%, 20–30% annualized return on capital deployed. That is the actual goal. Anything claiming higher is overfit. The architectural ambition is to compound knowledge in the sessions table over years — that is the moat that does not exist in retail systems.

---

## 6. Immediate Next Steps (the first PR)

In rough order, these are the changes that unblock everything else:

1. **`infra/docker-compose.yml`**: switch postgres image to `timescale/timescaledb:latest-pg16`, add `redis:7`, add `qdrant/qdrant`. Pick **one** of {RabbitMQ, Kafka} — recommend RabbitMQ for simplicity at this stage.
2. **`migrations/`**: new revision creating the `sessions` hypertable (spec §8.7.2 schema, adapted: drop options-specific JSON keys), `universe_membership`, `earnings_calendar`, and a `system_state` row.
3. **`app/connectors/`**: new `polygon_client.py` (or Alpaca market-data fallback), `redis_client.py`. Move `dhanhq/` to `legacy/`.
4. **`app/pipelines/`**: ingest scripts for prices, macro (FRED), VIX, COT, earnings.
5. **`requirements.txt`**: add `lightgbm`, `mlfinlab`, `anthropic`, `qdrant-client`, `redis`, `alpaca-py`, `vectorbt`, `prometheus-client`. Remove RL / TF deps from the prod path (keep in `requirements-legacy.txt` if you want them around).
6. **`app/backtest/`**: skeleton event-driven backtester that runs the baseline strategy from §0's gate.
7. **Run the baseline gate.** If Sharpe < 0.5, fix data/execution before any ML work.

---

*Plan version 1.0. Adapts `final_requirements.txt` v1.6 (GLD options) to a multi-asset US equity + commodity-futures swing trading system with no options exposure. Iterate as the build reveals reality.*
