# CLAUDE.md — TraderWise

This file is loaded into Claude Code's context for every session in this repo.
The same conventions are mirrored in `.cursor/rules/*.mdc` for Cursor's auto
mode. Treat both as authoritative.

---

## What this project is

US stocks + commodities **swing trading** system (no options). Predict market
*conditions* (vol regime, RV expansion, direction, cross-sectional rank, pair
mean-reversion); compose them into named scenarios; gate every order through
a deterministic risk layer; remember every play in a sessions table that
becomes the system's prior for similar future setups.

The repo is currently **stub-complete**: every module/port/adapter has its
contract in place; concrete implementations land per `STUBS_TODO.md`.

Read first if you're not already familiar:

- `README.md` — dataflow overview
- `IMPLEMENTATION_PLAN.md` — pivot decisions and gap list
- `BACKTESTING.md` — engine + labels + CV
- `VALIDATION.md` — validation pyramid + dashboard pages
- `STUBS_TODO.md` — per-function acceptance criteria

---

## Non-negotiable rules

1. **Hexagonal boundary.** Domain code (`app/core/**`) imports only from
   `app/core/ports/**`. Concrete adapters live in `app/connectors/**` or
   `app/services/**`. Domain code never imports `redis`, `aio_pika`,
   `polygon`, `anthropic`, etc. directly — only via Protocols.
2. **No-math LLM.** Numbers in LLM responses must originate from a tool's
   structured output. `services/llm/guards.py:enforce_no_math` enforces this.
   When you write a tool, define input + output pydantic models in
   `services/llm/schemas.py`.
3. **Point-in-time correctness.** Reading past data goes through
   `app/core/ai/features/asof.py:asof_join` or `asof_lookup`. Never
   `df[df.ts < now]` in feature code.
4. **Parity rule.** `ScenarioManager`, `RiskGateway`, models and features run
   identically in live, paper and backtest. Only adapters differ. If you find
   yourself writing `if mode == "backtest"` inside `app/core/ai/**`, stop —
   that branch belongs in an adapter, not in domain code.
5. **Single risk chokepoint.** Every order goes through
   `RiskGateway.evaluate`. No exceptions, no "just for testing" bypass.
6. **Bus topics in one place.** `app/services/messaging/topics.py`. Never
   publish or subscribe with a literal string.
7. **Determinism in backtest.** `BacktestEngine.run` must be byte-identical
   across two runs of the same config. Seed all RNGs from `config.seed`. No
   wall-clock calls inside the loop.
8. **Walk-forward CV only.** Use `WalkForwardSplit`, `PurgedKFold`, or
   `CombinatorialPurgedKFold`. Never `sklearn.model_selection.KFold` on
   time-series data.
9. **Calibrated probabilities + Kelly cap.** Wrap classifier outputs with
   `IsotonicCalibrator` before sizing. Sizing = fractional Kelly (0.25×) AND
   absolute cap of 1% of equity per trade.
10. **Stubs raise `NotImplementedError`.** Never return a fake/empty value to
    "let things compile". The whole point of a stub is to fail loudly.

## Implicit "do not"

- Do not introduce options, Greeks, IV surfaces — pivot is cash + futures only.
- Do not introduce Kafka — RabbitMQ is the bus.
- Do not bring back Dhan / Indian markets — pivot is US.
- Do not use `pandas` in new feature/model code — `polars` for tabular,
  `numpy` for numeric arrays.
- Do not change a function signature without updating callers in the same PR.
- Do not raise `NotImplementedError` in a function that already has a working
  implementation.

---

## Where things live

```
app/core/ports/        Protocol contracts (the seam)
app/core/di/           Container.build(mode) wires adapters
app/core/ai/features/  registry · asof · transforms · feature_sets
app/core/ai/models/    A vol_regime · B′ rv_expansion · C direction · D cross_section · E commodity_pairs · meta_labeler
app/core/ai/scenarios/ trend_continuation · mean_reversion · vol_expansion · ScenarioManager
app/core/ai/risk/      sizing · blackouts · circuit_breaker · gateway
app/core/ai/training/  labels · splitters · calibration · metrics · harness
app/core/ai/session/   Session model + repository
app/core/ai/validation/ psi · ks · calibration · deflated_sharpe
app/connectors/        polygon · alpaca · ibkr · fred · redis · qdrant · rabbitmq · telegram · twilio
app/services/          llm · sentiment · rag · messaging · telegram · audit · monitoring · retrieval
app/backtest/          engine · replayer · frictions · policies · runners · tests
app/pipelines/         ingest_prices · macro · vix · cot · earnings · news · rag · universe
app/dashboard/         Streamlit Home + 10 pages
app/api/main.py        operator-only HTTP surface
app/main.py            long-running worker entrypoint
```

---

## Common workflows

**Add a scenario** → see `.cursor/rules/10-skill-add-scenario.mdc`. Implements
`Scenario` Protocol; emits `ScenarioSignal`; deterministic; tested at parity
across live/backtest.

**Add a feature** → see `.cursor/rules/11-skill-add-feature.mdc`. Pure
function in `transforms.py`; registered with `(name, version)`; bump version
on math change; unit-tested with pinned outputs.

**Add a model** → see `.cursor/rules/12-skill-add-model.mdc`. LightGBM
backbone; implement `BaseModel`; train via `TrainingHarness` with walk-forward
CV; calibrate before sizing.

**Add a pipeline** → see `.cursor/rules/13-skill-add-pipeline.mdc`. Idempotent
`async def run(asof, ...) -> int`; UPSERT on natural key; provider injected.

**Implement a stub** → look it up in `STUBS_TODO.md`; replace body; add unit
test in mirrored `tests/` path; `make typecheck && make lint && make test`.

---

## Tooling

| Command | What it does |
|---|---|
| `make up` | start docker stack (postgres+timescale, redis, qdrant, rabbitmq) |
| `make migrate` | `alembic upgrade head` |
| `make api` | uvicorn, operator surface |
| `make worker` | `python -m app.main` |
| `make dashboard` | streamlit |
| `make test` | pytest |
| `make typecheck` | pyright (config in `pyrightconfig.json`) |
| `make lint` | ruff check |
| `make format` | ruff format |

---

## When in doubt

- Default to writing **less** code. The skeleton is right; almost every change
  is a stub→implementation, not a new abstraction.
- Default to writing **no comments** unless WHY is non-obvious.
- Default to **reading `STUBS_TODO.md` for the module you're touching** before
  starting.
- Default to **asking** if a request would violate one of the non-negotiable
  rules — they exist because we got burned without them.
