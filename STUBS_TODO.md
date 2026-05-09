# STUBS_TODO — Per-function acceptance criteria for the pivot stubs

> **How to use this file.** Every `raise NotImplementedError` body in `app/` is listed
> below with a short todo and verifiable acceptance criteria. Implement one module at
> a time. After each module, run `make typecheck && make test`. Cross-references:
> `IMPLEMENTATION_PLAN.md` (IP), `BACKTESTING.md` (BT), `VALIDATION.md` (VAL).
>
> **Conventions.**
> - Every function below already has a stable signature; do not change it without
>   updating its dependents in the same PR.
> - All datetimes are tz-aware UTC unless the docstring says otherwise.
> - Bars/features use `polars` (`pl.DataFrame`); model interiors use `numpy`.
> - Network access in tests is forbidden; gate live calls on `settings.NO_NETWORK`.
> - Every adapter raises `NotImplementedError` until concrete; never silently return
>   empty results — that hides bugs.
> - For each implementation: write a unit test in `tests/<mirrored path>/`.

---

## 1. Domain ports (`app/core/ports/*`)

These are Protocols. There is **nothing to implement** here — they're the contract.
Touch only to add a method that ALL adapters need; if it's only one adapter, put it
on the adapter instead.

| File | Why it exists | Edit when |
|---|---|---|
| `market_data.py` | bars + quotes + live subscriptions | Adding a new asset class shape |
| `broker.py` | submit / cancel / positions / equity | Never (US equities + futures only) |
| `macro.py` | FRED series, COT, VIX term structure | Adding a new macro source family |
| `news.py` | recent headlines per symbol | Switching providers (still keeps shape) |
| `cache.py` | KV w/ TTL + atomic incr | Adding stream primitives (don't — use bus) |
| `vector_store.py` | upsert + similarity search | Switching from Qdrant to pgvector-only |
| `bus.py` | publish / subscribe by routing key | Never |
| `notifier.py` | telegram / sms / email send | Adding a channel |
| `llm.py` | tool-using chat completion | Adding a new LLM vendor |

---

## 2. Connectors (`app/connectors/*`)

### `polygon_client.py` — `PolygonMarketData`
- `__init__(api_key, base_url)` — store config, lazy-init aiohttp `ClientSession` (close on `aclose()`).
- `get_bars(symbol, start, end, timeframe)` — call `/v2/aggs/ticker/.../range/.../{from}/{to}`; parse to `Bar`; sort ascending; dedupe by ts; return list. Acceptance: identical row count for same window across two calls; tz=UTC.
- `get_latest_quote(symbol)` — `/v2/last/nbbo/{symbol}`; emit `Quote(bid, ask, bid_size, ask_size, ts)`.
- `subscribe_bars(symbols, timeframe)` — websocket `wss://socket.polygon.io/stocks`; auth, subscribe `AM.<sym>` or `A.<sym>`; yield each as `Bar`. Reconnect with exponential backoff (cap 60s). Acceptance: survives one forced disconnect; yields ≥1 bar.

### `alpaca_client.py` — `AlpacaMarketData` + `AlpacaBroker`
- MarketData mirrors Polygon shape but hits `data.alpaca.markets`.
- `AlpacaBroker.submit(order)` — POST `/v2/orders`; map `OrderType.LIMIT` → `limit`, `MARKET` → `market`; return Alpaca `id`. Acceptance: rejected orders raise `BrokerRejected` (define in `ports.broker`).
- `cancel(id)` — DELETE; idempotent (cancel of non-existent order returns silently).
- `get_position(symbol)` — GET `/v2/positions/{symbol}`; 404 → None.
- `list_positions()` — GET `/v2/positions`; map to `Position`.
- `get_account_equity()` — GET `/v2/account` → float `equity`.
- Honour `paper=True` switching `paper-api.alpaca.markets`.

### `ibkr_client.py` — `IBKRMarketData` + `IBKRBroker`
- Use `ib_insync`. `__init__` connects on first call; reuse connection across calls.
- `get_bars` — `reqHistoricalData` with `useRTH=True`, `formatDate=2`; convert tz to UTC.
- `subscribe_bars` — `reqRealTimeBars(5)` for 5-sec bars, then aggregate to requested timeframe in-process.
- Broker side: futures only. Map symbols `"GC"`/`"CL"` etc. to `Future` with `lastTradeDateOrContractMonth` set to **front-month rolled** (BT §4.4).
- `get_account_equity` — `accountSummary().NetLiquidation`.

### `fred_client.py` — `FredClient`
- `get_series(series_ids, start, end)` — `https://api.stlouisfed.org/fred/series/observations`; one call per id; return `dict[str, pl.DataFrame(date, value)]`. Skip `"."` rows. Cache to `cache` port keyed by `(id, start, end)` for 24h.
- `get_cot_report(symbols, start, end)` — pull CFTC Commitments of Traders (FinancialFutures + Disaggregated as needed); return DataFrame with cols `as_of, symbol, comm_long, comm_short, noncomm_long, noncomm_short`.
- `get_vix_term_structure(as_of)` — derive `{VIX, VIX9D, VIX3M, VIX6M, VIX/VIX3M}` from latest values; raise if any missing.

### `redis_client.py` — `RedisCache`
- Wrap `redis.asyncio.Redis.from_url(url)`.
- `set(key, value, ttl_seconds)` — pass `ex=ttl_seconds` if not None; bytes only.
- `incr(key, ttl_seconds)` — `INCR` then `EXPIRE` if first-time set (use a Lua script to make it atomic, or `SET NX EX` then `INCR`).

### `qdrant_client.py` — `QdrantVectorStore`
- Use `qdrant_client.AsyncQdrantClient(host, port=port, prefer_grpc=True)`.
- `upsert(collection, ids, vectors, payloads)` — auto-create collection w/ size=len(vectors[0]) and Cosine distance if missing.
- `search(collection, query_vector, k, filter)` — return list of `VectorHit(id, score, payload)`. Filter dict → `qdrant_client.http.models.Filter`.

### `rabbitmq_client.py` — `RabbitMQBus`
- Use `aio-pika`.
- `__init__` declares one **topic exchange** named `"traderwise.events"`.
- `publish(topic, payload, headers)` — JSON-encode payload; set `delivery_mode=PERSISTENT`; `routing_key=topic`.
- `subscribe(topic_pattern, queue_name)` — declare durable queue; bind to exchange with pattern; `async for` yield `BusMessage(topic, payload, headers, delivery_tag)`. Acknowledge ONLY after consumer returns (caller controls `ack` via context manager — add one).

### `telegram_client.py` — `TelegramNotifier`
- POST `https://api.telegram.org/bot{token}/sendMessage` with `chat_id`, `text`, optional `reply_markup` (inline keyboard for HITL approvals).
- `parse_mode="MarkdownV2"`; escape per Telegram rules.

### `twilio_client.py` — `TwilioNotifier`
- Use `twilio.rest.Client`. `send(channel, recipient, body)` posts SMS; raise on failure.

---

## 3. Features (`app/core/ai/features/*`)

### `registry.py` — `FeatureRegistry`
- `__init__` — empty in-memory dict keyed by `(name, version)`.
- `register(spec)` — compute `body_hash = sha256(spec.fn.__code__.co_code).hexdigest()[:16]`; reject if `(name, version)` exists with a different body_hash.
- `get(name, version=None)` — return latest version if `None`; raise `KeyError` otherwise.
- `build_set(names_versions, df)` — call each spec.fn with the same input df; concat horizontally, prefix col names with `<name>__v<version>__`. Return `pl.DataFrame`. Acceptance: column order deterministic.
- `get_registry()` — module-level singleton.

### `asof.py` — point-in-time helpers (BT §4.1)
- `asof_join(left, right, on, by, allow_exact_matches=False, tolerance=None)` — wrapper over `pl.DataFrame.join_asof` with strategy `"backward"`. Forbid forward joins (raise). Used to attach lagged macro / news to bars without leaking future.
- `asof_lookup(series, ts)` — return last row with `ts_index <= ts`; None if none.

### `transforms.py`
For every function below: input is a polars Series (or DataFrame for `cross_sectional_rank`); output same length; first `window-1` rows NaN.
- `log_return(close, periods=1)` — `log(close / close.shift(periods))`.
- `realized_vol(close, window)` — std of log returns over window, annualised by `sqrt(252)`.
- `parkinson_vol(high, low, window)` — `(1/(4*ln 2)) * mean(ln(H/L)^2)` over window, sqrt-annualised.
- `atr(high, low, close, window=14)` — true range (max of H-L, |H-Cprev|, |L-Cprev|), then EMA over `window`.
- `rsi(close, window=14)` — Wilder's RSI.
- `zscore(s, window)` — `(s - rolling_mean) / rolling_std`.
- `rolling_rank_pct(s, window)` — percentile rank within window.
- `cross_sectional_rank(df, value_col, group_col)` — within each group (e.g. ts), rank value_col percentile-wise across symbols.

### `feature_sets.py`
Already declares the constants. Implementation = bind each name to a registered `FeatureSpec` in `registry`. Acceptance: `FeatureRegistry.build_set(A_VOL_REGIME, bars_df)` returns a DataFrame with the documented columns.

---

## 4. Sessions (`app/core/ai/session/*`)

### `models.py`
Already complete (dataclass mirror of schema). When schema changes, update both.

### `repository.py` — `SessionRepository` (Protocol)
Implement against `migrations/versions/a1b2c3d4e5f6_core_schema_pivot.py` schema in a concrete `app/core/repositories/session_pg.py` (create new):
- `create(session)` — INSERT all columns. `birth_embedding` cast to `vector(768)`.
- `update(session)` — UPDATE by `id`; bump `updated_at`.
- `get(id)` — SELECT *; return None on miss.
- `list_open(symbol, scenario, mode)` — `WHERE status='open'` + optional filters; ORDER BY `opened_at DESC`.
- `close(id, outcome, pnl, closed_at)` — single UPDATE setting `status='closed'`, `outcome`, `realized_pnl_usd`, `closed_at`, `updated_at = now()`.
- `find_neighbours(embedding, k, filters)` — `ORDER BY birth_embedding <=> $1 LIMIT k` (cosine), respecting `mode`/`scenario` filters. Acceptance: returns rows with `distance` field added.

---

## 5. Scenarios (`app/core/ai/scenarios/*`)

### `base.py` — Protocol; nothing to implement.

### `manager.py` — `ScenarioManager`
- `__init__(scenarios)` — store as list, build `dict[symbol → list[Scenario]]` via `applicable_universe`.
- `register(scenario)` — append; rebuild index.
- `applicable(symbol)` — O(1) lookup from index.
- `on_candle(ctx)` — for each applicable scenario for `ctx.symbol`: call `scenario.on_candle(ctx)`; collect non-None signals into a flat list; return. **Must be deterministic given identical ctx.** Acceptance: same instance reused in live and backtest (BT §1, IP §3.5).

### `trend_continuation.py` — daily; long when 50d > 200d AND price within 5% of 20d high AND ADX(14) > 25.
- `applicable_universe()` — return stored list.
- `on_candle(ctx)` — read precomputed features from `ctx.features` (no math). Emit `ScenarioSignal(SignalKind.OPEN, side=long, size_hint=1.0, features={...})` when condition holds. Else None.

### `mean_reversion.py` — daily; rsi2 < 5 AND price > 200d MA AND ATR-normalised distance from 20d MA > 1.0σ.

### `vol_expansion.py` — daily; realized_vol_5 / realized_vol_21 > 1.5 AND VIX term structure inverted (VIX/VIX3M > 1.0).

For all three: write a unit test that constructs a synthetic `ctx` and asserts the signal kind + side, no real data.

---

## 6. Models (`app/core/ai/models/*`)

All implement the `BaseModel` Protocol. Backbone = LightGBM unless noted.

### `vol_regime.py` — `VolRegimeModel` (Model A; multi-class: low/mid/high)
- `fit(X, y, sample_weight, eval_set)` — `lgb.LGBMClassifier(objective="multiclass", num_class=3, **params)`; train; store `feature_importances_`.
- `predict_proba(X)` — `(n, 3)` array.
- `save(path)` — pickle the booster + feature schema; return `ModelArtifact(path, model_type, feature_hash, fit_metadata)`.
- `load(artifact)` — reconstruct.

### `rv_expansion.py` — `RVExpansionModel` (Model B′; binary)
Same shape as A but `objective="binary"`; threshold via `expansion_k` at label time (already in signature).

### `direction.py` — `DirectionModel` (Model C; binary 5d-ahead up/down)
Binary LGBM; expose `horizon_days` so it's reflected in feature window.

### `cross_section.py` — `CrossSectionRankModel` (Model D; LambdaRank)
- `lgb.LGBMRanker(objective="lambdarank", **params)`.
- `fit` requires a `group` column passed via `sample_weight` slot or new kwarg — extend the protocol's signature with `groups: np.ndarray | None = None`.
- `predict_proba` returns ranking scores normalised by softmax across each ts group; emit top_k as `1.0`, others as `0.0` (binary "selected"). Acceptance: top_k count per ts equals `self.top_k`.

### `commodity_pairs.py` — `CommodityPairsModel` (E; OU)
- Fit OLS hedge ratio between the two legs over training window; compute spread; fit Ornstein-Uhlenbeck via Maximum Likelihood (or simple AR(1) with mean reversion proxy).
- `predict_proba` returns prob that spread mean-reverts within horizon (use OU half-life and current z-score).
- Acceptance: half-life > 0 and < 60 trading days else raise `UnstablePairError`.

### `meta_labeler.py` — gates primary signals
- LGBM binary on features + primary signal probability + scenario kind one-hot.
- Output: probability that taking the trade earns positive triple-barrier outcome.
- Used downstream to scale Kelly size.

---

## 7. Training (`app/core/ai/training/*`)

### `labels.py`
- `triple_barrier(events, prices, pt_sl_atr, max_holding_bars)` — for each `event_ts`, walk forward up to `max_holding_bars`; first barrier hit wins. Return `pl.DataFrame(event_ts, hit_barrier{up,down,time}, ret, t1)`. (BT §6.1)
- `meta_labels(primary_signals, prices, ...)` — derive `0/1` from triple-barrier `ret` sign. (BT §6.2)
- `sample_weights_by_uniqueness(events, t1)` — Lopez de Prado's avg-uniqueness; return weights array. Acceptance: weights ∈ (0,1], sum-normalisation optional.

### `splitters.py`
- `WalkForwardSplit.split` — yield `Fold(train_idx, test_idx)` rolling forward by `step`; train length capped at `train_size` if set.
- `PurgedKFold.split(df, t1)` — k-fold on time-ordered df; for each test fold, **purge** training rows whose `t1` falls inside the test window, then **embargo** an `embargo_pct` slice on either side. (BT §6.3)
- `CombinatorialPurgedKFold.split` — generate all `C(n_splits, n_test_splits)` combinations; same purge+embargo. Acceptance: every train fold disjoint from test in time domain.

### `calibration.py`
- `IsotonicCalibrator.fit(p_raw, y)` — `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`.
- `transform(p_raw)` — apply.
- `reliability_curve(p, y, n_bins=10)` — equal-frequency bins; return `(bin_centres, mean_p, mean_y)`.
- `brier_score(p, y)` — `mean((p - y)**2)`.

### `metrics.py`
- `sharpe(returns, ppy)` — `mean / std * sqrt(ppy)`.
- `deflated_sharpe(sharpe_obs, n_trials, skew, kurt, n_obs)` — Bailey & Lopez de Prado formula. (BT §6.5, VAL §L3)
- `probabilistic_sharpe_ratio(sharpe_obs, sharpe_ref, n_obs, skew, kurt)`.
- `max_drawdown(equity)` — running max minus current, divided by running max; return min.
- `hit_rate(returns)` — `mean(returns > 0)`.
- `profit_factor(returns)` — `sum(positive) / abs(sum(negative))`; return `inf` if no negatives.

### `harness.py` — `TrainingHarness`
- `run(X, y, t1, splitter)` — for each fold: instantiate model via factory; fit on train; predict on test; collect predictions, fold metrics. Return `dict[fold → metrics]` + concatenated out-of-fold predictions.
- `evaluate(model, X_test, y_test)` — return `{sharpe, brier, hit_rate, calibration_error}` for the test set.

---

## 8. Risk (`app/core/ai/risk/*`)

### `sizing.py`
- `kelly_fraction(p_win, win_loss_ratio)` — `(p_win * win_loss_ratio - (1 - p_win)) / win_loss_ratio`; floor at 0.
- `position_size_shares(SizingInputs)` — apply fractional Kelly (`kelly * 0.25`), cap at `max_pct_equity` (default 0.01), convert to whole shares using `entry_price` and `equity_usd`. Acceptance: result × entry_price ≤ 1% × equity_usd.

### `blackouts.py` — `EventBlackout`
- `__init__(windows)` — store interval tree (sorted list, bisect lookup OK).
- `is_blocked(symbol, ts)` — return matching window or None.
- `from_earnings_calendar(rows, days_before, days_after)` — for each earnings row produce `BlackoutWindow(symbol, start=announce - days_before, end=announce + days_after, kind="earnings")`. (IP §1.6)

### `circuit_breaker.py` — `CircuitBreaker`
- `evaluate(equity_curve, recent_trades)` — return matching `BreakerTrip` enum or None; rules: daily DD > 2%, weekly DD > 5%, ≥3 consecutive losing trades.
- `trip(reason)` — write to `system_state.trading_halted=true`, `halt_reason=reason`; emit `RISK_CIRCUIT` bus event.
- `reset(operator)` — set `trading_halted=false`, log `system_state.last_resumed_by`, emit `RISK_CIRCUIT(resumed)`.

### `gateway.py` — `RiskGateway`
- `evaluate(intent, account, position, blackouts, breaker)` — single chokepoint. Order of checks (return first failure):
  1. `breaker.tripped` → `RiskRejection.CIRCUIT_OPEN`.
  2. `blackouts.is_blocked(symbol, ts)` → `RiskRejection.EVENT_BLACKOUT`.
  3. existing position direction conflict → `RiskRejection.OPPOSITE_OPEN_POSITION`.
  4. sector / portfolio-beta cap → `RiskRejection.PORTFOLIO_LIMIT`.
  5. sizing > max → `RiskRejection.SIZING_CAP`.
  Else `RiskDecision.allow(size_shares=...)`. **No trade reaches the broker without going through this.** (IP §3.7)

---

## 9. LLM (`app/services/llm/*`)

### `anthropic_client.py` — `AnthropicLLMClient`
- Wrap `anthropic.AsyncAnthropic`.
- `complete(messages, tools, ...)` — tool-use loop **lives in orchestrator, not here**. This client only does one round-trip.
- Map `Tool` Protocol → Anthropic `tools` schema. Map `tool_use` blocks back into `LLMResponse.tool_calls`.

### `schemas.py`
Already pydantic models. **Do not let the LLM compute numbers**: every numeric field in an `*Out` schema must come from a tool's structured output, never the LLM body.

### `tools.py`
- `_get_market_snapshot(payload)` — read latest features for `payload['symbols']` from cache + DB; return `GetMarketSnapshotOut` JSON.
- `_score_scenario(payload)` — run `ScenarioManager.on_candle` synchronously over the supplied snapshot; return signals.
- `_propose_plan(payload)` — combine signals + risk gateway dry-run to return a candidate plan; **does not place orders**.
- `_search_similar_sessions(payload)` — embed `payload['birth_state_text']` via Bge → `SessionRepository.find_neighbours`.
- `default_toolset()` — list of `make_tool(name, description, schema, callable)` for the four above.

### `routing.py` — `pick_model`
- Tier mapping: `TaskKind.PARSE → haiku-4-5`, `TaskKind.PLAN → sonnet-4-6`, `TaskKind.RISK_REVIEW → opus-4-7`. If `no_network=True`, raise `LLMUnavailable`.

### `guards.py`
- `redact_pii(text)` — strip phone, email, SSN-like patterns.
- `looks_like_jailbreak(text)` — substring match a small denylist; OR call moderation API in the future.
- `validate_output(raw, schema)` — `schema.model_validate(raw)`; raise on failure.
- `enforce_no_math(text, allowed_numbers)` — extract every number from `text`; assert each appears in `allowed_numbers` (i.e. came from a tool). (IP §5.4)

### `orchestrator.py` — `LLMOrchestrator`
- `run(task, kind)` — pick model; loop ≤ `MAX_TOOL_HOPS=8`:
  1. `client.complete(messages, tools)`.
  2. If response has tool_calls → execute each via `tools[name].callable(args)`; append tool result to messages; continue loop.
  3. Else → call `enforce_no_math(text, allowed_numbers_from_tool_outputs)`; `validate_output`; return.
- Acceptance: deterministic tool-call order across two runs with identical input.

---

## 10. Other services (`app/services/*`)

### `sentiment/finbert_scorer.py` — `FinBertScorer`
- `__init__` — lazy-load HF transformers pipeline `text-classification` with model `ProsusAI/finbert`; pin device cpu unless GPU env var set.
- `score_batch(items)` — return `[{positive, negative, neutral}]` aligned with input order; batched (`batch_size=32`).

### `sentiment/news_volume.py` — `NewsVolumeScorer`
- `zscore(symbol, ts, recent_count)` — fetch `baseline_days` of headline counts from DB; return `(recent_count - mean) / std`. NaN when std=0.

### `rag/embedder.py` — `BgeEmbedder`
- `__init__` — load `BAAI/bge-base-en-v1.5` via sentence-transformers.
- `embed(texts)` — return `len(texts) × 768` float lists; normalise to unit length (cosine-friendly).

### `rag/indexer.py` — `RagIndexer`
- `index_documents(docs)` — embed → upsert into `VectorStore`; payloads carry `{source, ts, symbol}`.

### `rag/retriever.py` — `RagRetriever`
- `search(query, k, filters)` — embed query; `store.search(...)`; return list of `(text, score, payload)`.

### `messaging/topics.py`
Already a constants module. Treat as the single source of truth. Anyone publishing or subscribing must import from here, never literal strings.

### `messaging/publisher.py` — `EventPublisher`
- For each method: `await self.bus.publish(topic=topics.<NAME>, payload=payload, headers={"schema":"v1"})`.

### `messaging/consumer.py` — `BaseConsumer`
- `handle(msg)` — abstract; subclasses override.
- `run_forever()` — subscribe to `self.topic_pattern` (subclass attribute), call `handle`, ack on success, nack-requeue on exception. Add a circuit-break-after-N-failures guard.

### `telegram/approval_flow.py` — `TelegramApprovalFlow`
- `request_approval(card)` — generate `request_id=uuid4()`; cache `card` under `f"hitl:{request_id}"` with TTL = `timeout_seconds`; `notifier.send(...)` with inline Yes/No keyboard; create `asyncio.Future`; store in `_pending`; await with `asyncio.wait_for`. On timeout → return `Decision.TIMEOUT` (treated as "no").
- `handle_callback(request_id, choice, operator)` — set the matching Future's result.

### `telegram/daily_brief.py` — `DailyBriefSender`
- `send(as_of, brief)` — render Markdown summary (open sessions, P&L, breaker state, top signals) and call `notifier.send`.

### `audit/sink.py` — `AuditSink`
- `write(event)` — append-only JSONL with `event.id`, `ts`, `kind`, `payload`. Use `aiofiles`. fsync every 10 events.
- `query(start, end, kind)` — read whole file (small for v1), filter; return list. Acceptance: order preserved.

### `monitoring/metrics.py` — `MetricsRegistry`
- Singleton: `__init__` private-ish; `init()` registers Prometheus collectors:
  `bars_ingested_total`, `signals_emitted_total`, `orders_submitted_total`, `orders_rejected_total`, `breaker_trips_total`, `model_latency_ms` (Histogram), `feature_build_ms` (Histogram).
- `get_metrics()` returns the singleton.

### `monitoring/logging.py`
- `configure_logging(level, json_output)` — wire `structlog` with `JSONRenderer` if `json_output` else `ConsoleRenderer`.

### `retrieval/session_search.py` — `SessionSearch`
- `by_birth_state(text, k, filters)` — embed `text` via `BgeEmbedder`; call `sessions.find_neighbours`. Returns sessions with their realised outcomes — used as a prior over scenario probabilities (IP §3.5).

---

## 11. Backtest (`app/backtest/*`)

### `engine.py` — `BacktestEngine`
- `run(config)` — orchestrates: build `BarReplayer(config.source_path)` → for each bar: feed to `ScenarioManager` → emit signals → `RiskGateway.evaluate` → `SimBroker.submit` → settle fills next bar → mark-to-market → record. Return `BacktestResult` with equity curve + trade log + per-fold metrics.
- Deterministic: `random.seed(config.seed)`; **no wall-clock calls inside the loop**.
- Acceptance: two runs on the same config produce byte-identical `BacktestResult.trades`.

### `replayer.py` — `BarReplayer`
- `stream(start, end, symbols)` — read parquet/duckdb shards; for each ts inner-join `universe_membership` so a symbol only appears on dates it was a constituent (BT §4.2). `yield` bars in (ts, symbol) order. Acceptance: replaying identical inputs yields identical sequence.

### `frictions/slippage.py`
- `EquitySlippage.cost_per_share(order, bar_volume, bar_close)` — `k * bar_close * sqrt(order.qty / bar_volume)`. Acceptance: monotonic in qty; zero when qty=0.
- `FuturesSlippage.cost_per_share(order, bar_volume, bar_close)` — `n_ticks(order.symbol) * tick_size(order.symbol)` minimum, plus participation term.

### `frictions/commissions.py` — `CommissionModel`
- `cost(order, fill_price, asset_class)` — equities: `max(per_order_min, per_share * qty)`; futures: `per_contract * qty`. Defaults from constructor.

### `frictions/fills.py` — `FillModel`
- `attempt_fill(order, bar, queue_position=None)`:
  - MARKET: fill at `bar.open` (or `bar.close` if config says so) ± slippage + commission.
  - LIMIT: only if bar's range crosses limit price; fill at limit (worst-case prudence).
  - Return `Fill(order_id, qty, price, ts, fees)` or None.

### `frictions/sim_broker.py` — `SimBroker`
- Implements `BrokerAdapter` purely in-memory.
- `submit(order)` — assign `id`; queue for next bar's fill attempt.
- `cancel(id)` — remove from queue; idempotent.
- `get_position`, `list_positions`, `get_account_equity` — return current marked-to-market state.
- Acceptance: parity test (BT §6) — running `SimBroker` against recorded paper-trade fills reproduces equity within ±1 cent per share traded.

### `policies/base.py`
Already a frozen dataclass. Add concrete policies in `policies/deterministic_policy.py` (BT §7) — not yet stubbed; create as needed.

### `runners/single.py` — `run_single(config, policy)`
- Build engine, attach policy, call `engine.run(config)`; return `BacktestResult`.

### `runners/walkforward.py` — `run_walkforward(config, policy, splitter)`
- For each fold from `splitter.split(...)`: train models (delegate to `TrainingHarness`); produce a fitted policy; backtest on test fold; concatenate.

### `runners/sweep.py` — `run_sweep(configs, policy_factory)`
- Run each config; collect Sharpe trials; apply `deflated_sharpe(...)` correction (BT §6.5). Return ranked DataFrame with deflated stats.

### `tests/__init__.py` — TODO
Implement these tests (each in its own file under `app/backtest/tests/`):
- `test_no_lookahead.py` — assert that for every feature, `feature_built_at <= bar.ts`. (BT §6.6)
- `test_parity.py` — paper trade vs. backtest on same window must agree on direction and magnitude within tolerance (VAL §L4).
- `test_survivorship.py` — assert delisted symbols disappear from replay on/after delisting date.
- `test_friction_sanity.py` — slippage, commissions, fills are non-negative; total cost monotonic in qty.

---

## 12. Pipelines (`app/pipelines/*`)

Each `run(asof, ...)` is a daily batch. Returns row count written. All must be **idempotent** (`UPSERT` on `(symbol, ts)` or unique key) so a re-run on the same `asof` overwrites.

### `ingest_prices.py` — `run(asof, symbols, provider, db)`
- For each symbol: `provider.get_bars(symbol, last_ts+1d, asof, "1d")` → upsert into `prices_eod` table. Validate OHLC sanity (high ≥ open/close, low ≤ open/close).

### `macro.py` — `run(asof, provider, db)`
- Pull these series: `DGS10`, `DGS2`, `T10Y2Y`, `DFF`, `DTWEXBGS`, `CPIAUCSL`, `PPIACO`, `UNRATE`, `PAYEMS`. Upsert.

### `vix.py` — `run(asof, provider)`
- `provider.get_vix_term_structure(asof)` → upsert into `vix_term` (asof, vix, vix9d, vix3m, vix6m, vix_v3m_ratio).

### `cot.py` — `run(asof, provider, symbols)`
- COT report for `GOLD, SILVER, COPPER, WTI_CRUDE, CORN, WHEAT, SOYBEANS, NATURAL_GAS, SP500_EMINI`; upsert into `cot_weekly`.

### `earnings.py` — `run(asof, lookahead_days)`
- Pull next 14 days from a vendor (Polygon "/vX/reference/dividends" or Finnhub if added later); upsert into `earnings_calendar`. Used by `EventBlackout.from_earnings_calendar`.

### `news.py` — `run(asof, provider, symbols, scorer)`
- Fetch headlines via `news.NewsProvider.fetch_recent`; score via `FinBertScorer`; persist to `news_items` with sentiment.

### `rag.py` — `run(asof, indexer, sources)`
- Re-index any new docs from `sources` (10-K excerpts, broker research) via `RagIndexer`.

### `universe.py` — `run(asof, universes)`
- Reload index constituents (sp500, russell2000, commodities_core); upsert into `universe_membership` so backtest replay can join on PIT membership (BT §4.2).

---

## 13. API (`app/api/main.py`)

Wire each endpoint to the corresponding domain singleton (held by `Container`):
- `/healthz` — already returns ok.
- `/readyz` — call `cache.get("__ping__")`, `bus.publish` no-op, `db.execute("SELECT 1")`. 200 if all OK; 503 with details else.
- `/metrics` — `prometheus_client.generate_latest()`.
- `/sessions`, `/sessions/{id}` — delegate to `SessionRepository`.
- `/system/halt` (auth required, operator role) — `circuit_breaker.trip(BreakerTrip.MANUAL)`.
- `/system/resume` — `circuit_breaker.reset(operator)`.
- `/backtest` — enqueue a backtest job (write a row to `backtest_runs`, send a `BUS.BACKTEST_REQUEST` event); return `run_id`.
- `/backtest/{run_id}` — read the row + return latest summary.

Acceptance: no trading-logic endpoint exists; every state-changing endpoint requires operator auth (add `Depends(operator_auth)`).

---

## 14. Worker entrypoint (`app/main.py`)

`async def main()` should:
1. Parse args (mode = live / paper / backtest, dry_run flag).
2. `container = Container.build(mode)`.
3. Start `MetricsRegistry`, `configure_logging`.
4. Spawn long-running tasks via `asyncio.gather`: one consumer per topic (signals_consumer, orders_consumer, hitl_consumer, audit_consumer); the live `MarketDataProvider.subscribe_bars` loop; the cron-style scheduler that fires daily pipelines at the right wall-clock times.
5. Trap `SIGINT/SIGTERM`; gracefully cancel and `await client.aclose()` on every connector.

Acceptance: `python -m app.main --mode paper --dry-run` exits 0 and prints a startup banner showing every wired adapter.

---

## 15. DI container (`app/core/di/container.py`)

`Container.build(mode)`:
- `LIVE`: real Polygon + Alpaca/IBKR + Anthropic + RabbitMQ + Redis + Qdrant.
- `PAPER`: Polygon (real) + Alpaca paper + Anthropic + RabbitMQ + Redis + Qdrant.
- `BACKTEST`: `ParquetMarketData` (read-only, BT §4) + `SimBroker` + `NoopNotifier` + in-memory `KVCache` + in-memory `MessageBus` (deterministic ordering).

Wire `RiskGateway`, `ScenarioManager`, `LLMOrchestrator`, `EventPublisher` from those primitives. Acceptance: every field on the returned `Container` is non-None.

---

## 16. Dashboard (`app/dashboard/*`)

Each page's `main()` is an empty stub. Wire each one to the spec in VAL §3.2:

| Page | Reads from | Renders |
|---|---|---|
| `Home.py` | sessions, metrics | KPI tiles + nav |
| `01_Data_Health` | DB | freshness + null heatmap |
| `02_Feature_Explorer` | parquet store | distribution + asof timeline |
| `03_Model_Performance` | model registry | reliability curves, confusion |
| `04_Backtest_Viewer` | `BacktestResult` | equity curve, trade table |
| `05_Scenario_Inspector` | sessions | scenario fire-rate, win-rate |
| `06_Session_Replay` | sessions | tick-by-tick replay of one session |
| `07_Trade_Journal` | broker fills | filterable journal |
| `08_Risk_Monitor` | breaker + gateway state | live limits |
| `09_Paper_vs_Backtest` | both | side-by-side diff |
| `10_Live_Overview` | bus + sessions | real-time tape |

Use `st.cache_data` aggressively; **never** call live brokers from the dashboard.

---

## 17. Validation (`app/core/ai/validation/*`)

Already minimal: `psi.py`, `ks.py`, `calibration.py`, `deflated_sharpe.py`. Each function above has a one-line stub. Implement per VAL §L1–L3. These functions are pure-numpy and must have unit tests with synthetic inputs (no DB).

---

## Definition of done — for each module above

1. Stub body replaced with implementation.
2. Unit test added at `tests/<mirrored path>/test_<module>.py`.
3. `make typecheck` clean (no new pyright errors in the module).
4. `make lint` clean for the touched files.
5. If the module crosses a port boundary, an end-to-end test under `app/backtest/tests/` exercises the whole path with `SimBroker` + in-memory cache + in-memory bus.
6. Public symbol added to module's `__init__.py` `__all__`.
