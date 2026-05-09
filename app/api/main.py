"""Operator-facing HTTP surface."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.core.di.container import Container, Mode
from app.core.ai.risk.circuit_breaker import BreakerTrip

app = FastAPI(title="TraderWise Operator API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_container: Container | None = None
_backtest_runs: dict[str, dict] = {}

security = HTTPBasic()


def get_container() -> Container:
    global _container
    if _container is None:
        _container = Container.build(Mode.PAPER)
    return _container


def operator_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    from app.config.settings import settings
    # Simple: any non-empty username works in dev; in prod wire to a secrets store
    if not credentials.username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, Any]:
    c = get_container()
    checks: dict[str, str] = {}
    ok = True
    try:
        await c.cache.get("__ping__")
        checks["cache"] = "ok"
    except Exception as e:
        checks["cache"] = str(e)
        ok = False
    try:
        await c.bus.publish("__ping__", {})
        checks["bus"] = "ok"
    except Exception as e:
        checks["bus"] = str(e)
        ok = False
    if not ok:
        raise HTTPException(status_code=503, detail=checks)
    return {"status": "ok", "checks": checks}


@app.get("/metrics")
async def metrics() -> Response:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/sessions")
async def list_sessions() -> list[dict]:
    c = get_container()
    if c.sessions is None:
        return []
    sessions = await c.sessions.list_open(symbol=None, scenario=None, mode=None)
    return [s.__dict__ if hasattr(s, "__dict__") else s for s in sessions]


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    c = get_container()
    if c.sessions is None:
        raise HTTPException(status_code=404, detail="Sessions not wired")
    session = await c.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.__dict__ if hasattr(session, "__dict__") else session


@app.post("/system/halt")
async def halt(reason: str = "manual", operator: str = Depends(operator_auth)) -> dict:
    c = get_container()
    if c.risk is None:
        raise HTTPException(status_code=503, detail="Risk gateway not wired")
    await c.risk.breaker.trip(BreakerTrip.MANUAL)
    return {"status": "halted", "reason": reason}


@app.post("/system/resume")
async def resume(operator: str = Depends(operator_auth)) -> dict:
    c = get_container()
    if c.risk is None:
        raise HTTPException(status_code=503, detail="Risk gateway not wired")
    await c.risk.breaker.reset(operator)
    return {"status": "resumed", "operator": operator}


@app.post("/backtest")
async def start_backtest(payload: dict) -> dict:
    run_id = str(uuid.uuid4())
    _backtest_runs[run_id] = {"status": "queued", "config": payload}
    if get_container().bus is not None:
        from app.services.messaging import topics
        await get_container().bus.publish(
            topics.AUDIT_EVENT,
            {"run_id": run_id, "config": payload},
            headers={"schema": "v1"},
        )
    return {"run_id": run_id, "status": "queued"}


@app.get("/backtest/{run_id}")
async def backtest_status(run_id: str) -> dict:
    run = _backtest_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
