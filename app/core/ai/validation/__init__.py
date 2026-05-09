"""Validation primitives — drift, calibration, distributional tests.

Per VALIDATION.md §2: every model artifact ships with a validation report;
production scoring continuously emits PSI + reliability + latency metrics so
the monitoring service can flag drift before PnL does.
"""
from .psi import psi_categorical, psi_continuous
from .ks import ks_test
from .calibration import reliability, brier
from .deflated_sharpe import deflated_sharpe

__all__ = [
    "psi_categorical", "psi_continuous",
    "ks_test",
    "reliability", "brier",
    "deflated_sharpe",
]
