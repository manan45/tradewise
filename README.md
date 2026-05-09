# TraderWise

> **US stocks + commodities swing-trading system. No options.**
> Predict market *conditions* (vol regime, RV expansion, direction, cross-sectional rank, pair mean-reversion); compose them into named scenarios; gate every order through a deterministic risk layer; remember every play in a sessions table that becomes the system's prior for similar future setups.

The repository is currently a **stub-complete skeleton**: every module, port, and adapter has its contract in place; concrete implementations land progressively per [`STUBS_TODO.md`](./STUBS_TODO.md).

---

## What this is, in one paragraph

TraderWise ingests EOD + intraday US equity bars (Polygon), futures bars (IBKR), macro series (FRED), VIX term structure, COT positioning and curated news. A feature registry pins every transform by `(name, version, body_hash)`. Five small models (vol regime, realized-vol expansion, direction, cross-sectional rank, commodity pairs) plus a meta-labeler convert features into calibrated probabilities. A `ScenarioManager` composes those probabilities into discrete *named* setups (trend continuation, mean reversion, vol expansion, …). Every candidate trade goes through a `RiskGateway` (Kelly-sized at 1% cap, event blackouts, circuit breaker). Approved orders go to Alpaca (equities) or IBKR (futures). Every trade opens a row in a `sessions` Timescale hypertable with a 768-d `birth_embedding` so future setups can find their nearest neighbours via pgvector. An LLM orchestrator narrates the why, but **never computes numbers** — it can only quote tool outputs.

Live, paper, and backtest run the **same** `ScenarioManager`/`RiskGateway` instances, swapping only the broker and market-data adapters via the DI container. Backtests use a deterministic event-driven engine with explicit slippage/commission/fill models, walk-forward + combinatorial purged k-fold CV, isotonic calibration, and deflated Sharpe.

---

## Dataflow

```
                                 ┌─────────────────────────────────────────────┐
                                 │ EXTERNAL                                    │
                                 │  Polygon · Alpaca · IBKR · FRED · CFTC ·   │
                                 │  News providers · Twitter (later)           │
                                 └─────────────────────┬───────────────────────┘
                                                       │
                              ┌────────────────────────▼────────────────────────┐
                              │ pipelines/ (idempotent daily batches)           │
                              │  ingest_prices · macro · vix · cot · earnings · │
                              │  news · rag · universe                          │
                              └────────────────────────┬────────────────────────┘
                                                       │
            ┌──────────────────────────────────────────┴──────────────────────────────────────┐
            │ STORAGE                                                                          │
            │  TimescaleDB (prices, macro, vix, cot, sessions, system_state, audit)           │
            │  pgvector(768) on sessions.birth_embedding                                       │
            │  Qdrant (RAG: news, research)                                                    │
            │  Redis (TTL state + idempotency)                                                 │
            └──────────────────────────────────────────┬──────────────────────────────────────┘
                                                       │
                          ┌────────────────────────────▼────────────────────────────┐
                          │ FEATURES (asof-correct)                                  │
                          │  features/registry → feature_sets:                        │
                          │   A_VOL_REGIME · B_PRIME_RV_EXPANSION · C_DIRECTION ·   │
                          │   D_CROSS_SECTION · E_COMMODITY_PAIRS · META             │
                          └────────────────────────────┬────────────────────────────┘
                                                       │
                          ┌────────────────────────────▼────────────────────────────┐
                          │ MODELS (LightGBM unless noted)                          │
                          │  A vol_regime · B′ rv_expansion · C direction ·         │
                          │  D cross_section (LambdaRank) · E commodity_pairs (OU) ·│
                          │  meta_labeler                                            │
                          │  trained via TrainingHarness (walk-forward + CPCV)       │
                          │  calibrated via IsotonicCalibrator                       │
                          └────────────────────────────┬────────────────────────────┘
                                                       │
                          ┌────────────────────────────▼────────────────────────────┐
                          │ ScenarioManager.on_candle(ctx)                          │
                          │  trend_continuation · mean_reversion · vol_expansion ·  │
                          │  (more in IMPLEMENTATION_PLAN.md §1.5)                  │
                          │  emits ScenarioSignal(OPEN/UPDATE/CLOSE)                │
                          └────────────────────────────┬────────────────────────────┘
                                                       │
       ┌───────────────────────────────────────────────┴───────────────────────────────────────────────┐
       │                                                                                               │
       │   ┌──────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────────┐ │
       │   │ LLMOrchestrator          │  │ SessionRepository       │  │ RiskGateway (CHOKEPOINT)    │ │
       │   │  4 tools, no math:       │  │  open / update / close  │  │  - circuit breaker          │ │
       │   │  get_market_snapshot     │  │  birth_embedding (768)  │  │  - event blackouts          │ │
       │   │  score_scenario          │  │  find_neighbours (pgv)  │  │  - position conflicts       │ │
       │   │  propose_plan            │  │                          │  │  - portfolio limits         │ │
       │   │  search_similar_sessions │  │                          │  │  - Kelly cap @ 1% equity    │ │
       │   └────────────┬─────────────┘  └─────────────┬───────────┘  └─────────────┬───────────────┘ │
       │                │                              │                            │                  │
       │                │     RAG (Qdrant)             │   Sessions table is        ▼                  │
       │                └─────► retriever ◄────────────┘   the system's memory   APPROVED              │
       │                                                                            │                  │
       └────────────────────────────────────────────────────────────────────────────┼──────────────────┘
                                                                                    │
                              ┌─────────────────────────────────────────────────────▼─────┐
                              │ EXECUTION                                                  │
                              │  Live  → Alpaca / IBKR                                     │
                              │  Paper → Alpaca paper                                      │
                              │  Backtest → SimBroker (deterministic, in-memory)           │
                              └─────────────────────────────────────────────────────┬─────┘
                                                                                    │
                              ┌─────────────────────────────────────────────────────▼─────┐
                              │ MESSAGE BUS  (RabbitMQ topic exchange "traderwise.events")│
                              │  topics in services/messaging/topics.py                    │
                              │  signals · orders · risk · circuit · sentiment · audit ·   │
                              │  hitl_request / hitl_response                              │
                              └─────────────┬─────────────────────────┬──────────────────┘
                                            │                         │
                          ┌─────────────────▼──────────┐  ┌───────────▼─────────────────┐
                          │ HITL (Telegram approval)   │  │ AuditSink (jsonl, append)   │
                          └────────────────────────────┘  └─────────────────────────────┘
```

**Parity rule.** `ScenarioManager`, `RiskGateway`, and the feature/model code paths run **identically** in live, paper and backtest. Only the broker + market-data + bus + cache adapters differ, swapped via [`app/core/di/container.py`](./app/core/di/container.py). This is what makes backtest results actionable.

**No-math LLM rule.** The LLM is a tool-using narrator. Every numeric value in its outputs must come from a tool's structured response — `services/llm/guards.py:enforce_no_math` rejects responses that introduce numbers the tools never returned.

**Sessions are memory.** Every opened trade writes a `Session` row with a `birth_embedding` describing the setup. When a new candidate setup arrives, `SessionSearch.by_birth_state` retrieves the nearest neighbours; their realised outcomes prior the meta-labeler.

---

## Repository layout

```
app/
  api/                FastAPI operator surface (healthz, sessions, halt/resume, backtest jobs)
  main.py             long-running worker entrypoint (consumers + scheduler + subscribe loops)
  config/settings.py  pydantic-settings (env / .env)
  connectors/         concrete adapters → Polygon · Alpaca · IBKR · FRED · Redis · Qdrant · RabbitMQ · Telegram · Twilio
  core/
    ports/            Protocol contracts every adapter must implement
    di/container.py   wires adapters by mode (LIVE | PAPER | BACKTEST)
    ai/
      features/       registry · asof helpers · transforms · feature_sets
      session/        Session model + Repository
      scenarios/      base + manager + (trend, mean-rev, vol-expansion stubs)
      models/         A · B′ · C · D · E · meta_labeler
      training/       labels · splitters · calibration · metrics · harness
      risk/           sizing · blackouts · circuit_breaker · gateway
      validation/     psi · ks · calibration · deflated_sharpe
  services/
    llm/              anthropic client · schemas · tools · routing · guards · orchestrator
    sentiment/        finbert scorer · news volume zscore
    rag/              embedder (BAAI/bge-base-en-v1.5) · indexer · retriever
    messaging/        topics · publisher · base consumer
    telegram/         approval flow · daily brief
    audit/sink.py     append-only JSONL
    monitoring/       prometheus metrics · structlog
    retrieval/        session_search.by_birth_state
  backtest/
    engine.py         event-driven, deterministic
    replayer.py       PIT join against universe_membership
    frictions/        slippage · commissions · fills · sim_broker
    policies/         policy contract; concrete policies as added
    runners/          single · walkforward · sweep (deflated Sharpe)
  pipelines/          ingest_prices · macro · vix · cot · earnings · news · rag · universe
  dashboard/          Streamlit Home + 10 pages (Data Health → Live Overview)
infra/                docker-compose (postgres-tsdb, redis, qdrant, rabbitmq) + Dockerfiles
migrations/           alembic; core_schema_pivot creates sessions / universe_membership / earnings_calendar / system_state
```

---

## Quickstart (local)

```bash
# 1. infra
make up                     # postgres+timescale, redis, qdrant, rabbitmq

# 2. schema
make migrate                # alembic upgrade head

# 3. processes (separate terminals)
make api                    # uvicorn app.api.main:app --reload (operator surface)
make worker                 # python -m app.main (consumers + scheduler)
make dashboard              # streamlit run app/dashboard/Home.py

# quality
make test
make lint
make typecheck
make format
```

Configure secrets via `.env`. See `app/config/settings.py` for the full env surface (Polygon / Alpaca / IBKR / FRED / Anthropic / Telegram / Twilio).

`NO_NETWORK=true` disables every outbound call — useful for unit tests and CI.

---

## Modes

| Mode | Market data | Broker | Bus | Cache | Notifier |
|---|---|---|---|---|---|
| `live` | Polygon (websocket) | Alpaca live / IBKR | RabbitMQ | Redis | Telegram + Twilio |
| `paper` | Polygon | Alpaca paper | RabbitMQ | Redis | Telegram |
| `backtest` | `ParquetMarketData` | `SimBroker` (in-memory) | in-process bus (deterministic) | in-memory | noop |

`Container.build(Mode.PAPER)` returns a fully wired `Container`; the same `ScenarioManager` and `RiskGateway` instances are used across modes.

---

## Documentation index

| Doc | What it covers |
|---|---|
| [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) | Pivot decisions, target system, gap-by-gap change list, phased roadmap |
| [`BACKTESTING.md`](./BACKTESTING.md) | Engine design, frictions model, labels, splitters, deflated Sharpe, no-look-ahead audit |
| [`VALIDATION.md`](./VALIDATION.md) | Validation pyramid (L1–L5), per-model checks, dashboard page specs |
| [`STUBS_TODO.md`](./STUBS_TODO.md) | Per-function acceptance criteria for every stub in the repo |
| [`CLAUDE.md`](./CLAUDE.md) | Repo conventions for AI assistants |
| [`.cursor/rules/`](./.cursor/rules/) | Cursor auto-mode rules mirroring CLAUDE.md |

---

## Status

Everything in `app/` is currently a stub. The right way to make progress is:

1. Pick a module from `STUBS_TODO.md`.
2. Implement the listed functions to spec.
3. Add unit tests in the mirrored `tests/` path.
4. Run `make test && make typecheck && make lint`.
5. Open a PR; the CI runs walk-forward backtest sanity checks.

License: MIT. See [LICENSE](./LICENSE).
