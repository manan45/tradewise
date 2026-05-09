"""TraderWise process entrypoint."""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from app.core.di.container import Container, Mode
from app.services.monitoring.logging import configure_logging
from app.services.monitoring.metrics import get_metrics


async def main() -> None:
    parser = argparse.ArgumentParser(description="TraderWise worker")
    parser.add_argument("--mode", choices=["live", "paper", "backtest"], default="paper")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--json-logs", action="store_true", default=False)
    args = parser.parse_args()

    configure_logging(level=args.log_level, json_output=args.json_logs)
    logger = logging.getLogger("traderwise.main")

    metrics = get_metrics()
    metrics.init()

    mode = Mode(args.mode)
    container = Container.build(mode)

    logger.info(
        "TraderWise starting",
        extra={
            "mode": mode.value,
            "dry_run": args.dry_run,
            "market_data": type(container.market_data).__name__ if container.market_data else "None",
            "broker": type(container.broker).__name__ if container.broker else "None",
            "bus": type(container.bus).__name__ if container.bus else "None",
            "cache": type(container.cache).__name__ if container.cache else "None",
            "llm": type(container.llm).__name__ if container.llm else "None",
        }
    )

    print(
        f"[TraderWise] mode={mode.value} dry_run={args.dry_run} "
        f"broker={type(container.broker).__name__} "
        f"bus={type(container.bus).__name__} "
        f"cache={type(container.cache).__name__}"
    )

    if args.dry_run:
        logger.info("Dry-run: exiting after startup.")
        return

    tasks: list[asyncio.Task] = []
    stop_event = asyncio.Event()

    def _handle_signal():
        logger.info("Shutdown signal received.")
        stop_event.set()
        for t in tasks:
            t.cancel()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    try:
        await stop_event.wait()
    finally:
        for t in tasks:
            t.cancel()
        # Gracefully close connectors
        for attr in ("market_data", "broker", "cache", "bus"):
            obj = getattr(container, attr, None)
            if obj is not None and hasattr(obj, "aclose"):
                try:
                    await obj.aclose()
                except Exception:
                    pass
        logger.info("TraderWise shut down cleanly.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
