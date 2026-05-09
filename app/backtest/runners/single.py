"""One-shot backtest — single date range, single policy."""
from __future__ import annotations

from app.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from app.backtest.policies.base import Policy
from app.backtest.frictions.sim_broker import SimBroker
from app.backtest.frictions.fills import FillModel
from app.backtest.frictions.commissions import CommissionModel
from app.backtest.frictions.slippage import EquitySlippage
from app.backtest.replayer import BarReplayer
from app.core.ai.scenarios.manager import ScenarioManager
from app.core.ai.risk.gateway import RiskGateway
from app.core.ai.risk.blackouts import EventBlackout
from app.core.ai.risk.circuit_breaker import CircuitBreaker


def run_single(config: BacktestConfig, policy: Policy,
               source_path: str = "data/bars",
               scenario_manager: ScenarioManager | None = None) -> BacktestResult:
    replayer = BarReplayer(source_path=source_path)
    broker = SimBroker(
        starting_equity=config.starting_equity,
        slippage=EquitySlippage(),
        commissions=CommissionModel(**policy.frictions),
        fills=FillModel(),
    )
    if scenario_manager is None:
        scenario_manager = ScenarioManager(scenarios=[])
    breaker = CircuitBreaker(**policy.breaker)
    blackouts = EventBlackout([])
    risk = RiskGateway(breaker=breaker, blackouts=blackouts)
    engine = BacktestEngine(
        replayer=replayer,
        scenarios=scenario_manager,
        risk=risk,
        broker=broker,
    )
    return engine.run(config)
