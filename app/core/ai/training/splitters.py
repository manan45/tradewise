"""Time-aware CV splitters — never use sklearn.KFold on financial data.

- WalkForwardSplit: expanding or rolling window, last fold is most recent.
- PurgedKFold: standard k-fold with a purge gap to prevent leakage from
  overlapping triple-barrier events.
- CombinatorialPurgedKFold: many train/test combinations to estimate the
  variance of the backtest statistic itself (deflated Sharpe input).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


class WalkForwardSplit:
    def __init__(self, n_splits: int, train_size: int | None = None,
                 test_size: int | None = None, gap: int = 0):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, df: pl.DataFrame, ts_col: str = "ts") -> Iterator[Fold]:
        n = len(df)
        test_sz = self.test_size or max(1, n // (self.n_splits + 1))
        # Build test fold start positions (rolling forward)
        test_starts = []
        for i in range(self.n_splits):
            start = n - (self.n_splits - i) * test_sz
            if start < 1:
                continue
            test_starts.append(start)

        for test_start in test_starts:
            test_end = min(test_start + test_sz, n)
            train_end = test_start - self.gap
            if train_end <= 0:
                continue
            train_start = 0 if self.train_size is None else max(0, train_end - self.train_size)
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            yield Fold(train_idx=train_idx, test_idx=test_idx)


class PurgedKFold:
    def __init__(self, n_splits: int, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, df: pl.DataFrame, t1: pl.Series) -> Iterator[Fold]:
        n = len(df)
        indices = np.arange(n)
        embargo = int(n * self.embargo_pct)
        fold_size = n // self.n_splits

        t1_list = t1.to_list()

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.n_splits - 1 else n
            test_idx = indices[test_start:test_end]

            # Purge: remove training rows whose t1 falls within the test window
            test_ts_start = df["ts"][test_start]
            test_ts_end = df["ts"][test_end - 1]

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start:test_end] = False
            # Embargo on both sides
            purge_start = max(0, test_start - embargo)
            purge_end = min(n, test_end + embargo)
            train_mask[purge_start:purge_end] = False

            # Purge rows by t1 overlapping test window
            for i in range(n):
                if train_mask[i] and t1_list[i] is not None:
                    if t1_list[i] >= test_ts_start and t1_list[i] <= test_ts_end:
                        train_mask[i] = False

            train_idx = indices[train_mask]
            if len(train_idx) == 0:
                continue
            yield Fold(train_idx=train_idx, test_idx=test_idx)


class CombinatorialPurgedKFold:
    def __init__(self, n_splits: int, n_test_splits: int = 2,
                 embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct

    def split(self, df: pl.DataFrame, t1: pl.Series) -> Iterator[Fold]:
        n = len(df)
        indices = np.arange(n)
        embargo = int(n * self.embargo_pct)
        fold_size = n // self.n_splits
        t1_list = t1.to_list()

        fold_ranges = []
        for fold in range(self.n_splits):
            start = fold * fold_size
            end = start + fold_size if fold < self.n_splits - 1 else n
            fold_ranges.append((start, end))

        for test_folds in combinations(range(self.n_splits), self.n_test_splits):
            test_indices = np.concatenate([
                indices[fold_ranges[f][0]:fold_ranges[f][1]] for f in test_folds
            ])
            test_ts_pairs = [(df["ts"][fold_ranges[f][0]], df["ts"][fold_ranges[f][1] - 1])
                             for f in test_folds]

            train_mask = np.ones(n, dtype=bool)
            for f in test_folds:
                s, e = fold_ranges[f]
                purge_s = max(0, s - embargo)
                purge_e = min(n, e + embargo)
                train_mask[purge_s:purge_e] = False

            for i in range(n):
                if not train_mask[i]:
                    continue
                if t1_list[i] is None:
                    continue
                for ts_start, ts_end in test_ts_pairs:
                    if t1_list[i] >= ts_start and t1_list[i] <= ts_end:
                        train_mask[i] = False
                        break

            train_idx = indices[train_mask]
            if len(train_idx) == 0:
                continue
            yield Fold(train_idx=train_idx, test_idx=test_indices)
