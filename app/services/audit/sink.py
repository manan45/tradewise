"""AuditSink — accepts AuditEvent, writes JSONL + emits to bus."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import aiofiles


@dataclass(frozen=True)
class AuditEvent:
    id: UUID = field(default_factory=uuid4)
    ts: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    kind: str = ""
    actor: str = "system"
    session_id: UUID | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class AuditSink:
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self._write_count = 0

    def _serialize(self, event: AuditEvent) -> str:
        return json.dumps({
            "id": str(event.id),
            "ts": event.ts.isoformat(),
            "kind": event.kind,
            "actor": event.actor,
            "session_id": str(event.session_id) if event.session_id else None,
            "payload": event.payload,
        })

    async def write(self, event: AuditEvent) -> None:
        os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
        async with aiofiles.open(self.jsonl_path, "a") as f:
            await f.write(self._serialize(event) + "\n")
            self._write_count += 1
            if self._write_count % 10 == 0:
                await f.flush()

    async def query(
        self,
        kind: str | None = None,
        session_id: UUID | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        if not os.path.exists(self.jsonl_path):
            return []
        results: list[AuditEvent] = []
        async with aiofiles.open(self.jsonl_path) as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = AuditEvent(
                    id=UUID(d["id"]),
                    ts=datetime.fromisoformat(d["ts"]),
                    kind=d["kind"],
                    actor=d["actor"],
                    session_id=UUID(d["session_id"]) if d.get("session_id") else None,
                    payload=d.get("payload", {}),
                )
                if kind is not None and event.kind != kind:
                    continue
                if session_id is not None and event.session_id != session_id:
                    continue
                if since is not None and event.ts < since:
                    continue
                results.append(event)
        return results[-limit:]
