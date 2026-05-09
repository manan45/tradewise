"""Feature registry — every feature is named, versioned, and hash-pinned.

Why: research code routinely renames or silently changes a feature; that
breaks production parity. The registry forces every consumer to look features
up by (name, version) and the loader hashes the definition body to detect
drift between training-time and live-time builds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import polars as pl


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: int
    inputs: tuple[str, ...]            # column names this feature needs
    compute: Callable[[pl.DataFrame], pl.Series]
    body_hash: str                     # sha256(inspect.getsource(compute))
    description: str = ""


class FeatureRegistry:
    def __init__(self) -> None:
        self._features: dict[tuple[str, int], FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        key = (spec.name, spec.version)
        existing = self._features.get(key)
        if existing is not None and existing.body_hash != spec.body_hash:
            raise ValueError(
                f"Feature ({spec.name}, v{spec.version}) already registered with a "
                f"different body_hash. Bump the version instead of changing the body."
            )
        self._features[key] = spec

    def get(self, name: str, version: int | None = None) -> FeatureSpec:
        if version is not None:
            key = (name, version)
            if key not in self._features:
                raise KeyError(f"Feature ({name}, v{version}) not registered.")
            return self._features[key]
        # Return the latest version
        matches = [v for (n, _v), v in self._features.items() if n == name]
        if not matches:
            raise KeyError(f"Feature '{name}' not registered.")
        return max(matches, key=lambda s: s.version)

    def build_set(
        self,
        df: pl.DataFrame,
        names: Iterable[tuple[str, int]],
    ) -> pl.DataFrame:
        """Compute and append all listed features to df, in dependency order."""
        result = df
        for name, version in names:
            spec = self.get(name, version)
            col = spec.compute(result)
            col_name = f"{spec.name}__v{spec.version}__{col.name}"
            result = result.with_columns(col.alias(col_name))
        return result


_REGISTRY: FeatureRegistry | None = None


def get_registry() -> FeatureRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = FeatureRegistry()
    return _REGISTRY
