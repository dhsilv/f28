"""
Minimal tick-replay engine. Placeholder for the live Strategy Studio
OnQuote entry point.

The user said: data parsing and tick schema live in a separate repo that
will eventually stream ticks directly into this process. This engine
therefore accepts two modes:

1. `attach_strategy(strategy)` + `run_stream(iterable_of_dicts)`
   -- for injected streams (recommended; matches the live Strategy
   Studio pattern).

2. `run_csv(path)` -- convenience fallback that reads a pre-parsed CSV
   in the schema expected by the strategy's on_tick contract. This is
   for sanity testing only; do NOT treat it as the canonical loader.
"""
from __future__ import annotations

import csv
from typing import Iterable, Optional

import numpy as np
import pandas as pd


class TickEngine:
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.strategy = None
        self._ticks_processed = 0

    def attach_strategy(self, strategy) -> None:
        self.strategy = strategy

    # ------------------------------------------------------------------
    def run_stream(self, tick_iterable: Iterable[dict]) -> None:
        if self.strategy is None:
            raise RuntimeError("attach_strategy() must be called before run_stream().")

        for tick in tick_iterable:
            self.strategy.on_tick(tick)
            self._ticks_processed += 1

        print(f"Stream complete. {self._ticks_processed} ticks processed.")

    def run(self) -> None:
        """Default entry: load CSV from self.data_path and replay it."""
        if self.data_path is None:
            raise RuntimeError("No data_path set. Use run_stream() for injected streams.")
        self.run_csv(self.data_path)

    def run_csv(self, path: str) -> None:
        if self.strategy is None:
            raise RuntimeError("attach_strategy() must be called before run_csv().")

        df = pd.read_csv(path, parse_dates=["timestamp", "f1_expiry", "f2_expiry"])
        for row in df.itertuples(index=False):
            tick = self._row_to_tick(row._asdict())
            self.strategy.on_tick(tick)
            self._ticks_processed += 1

        print(f"Replay complete. {self._ticks_processed} ticks processed.")

    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_tick(row: dict) -> dict:
        """Coerce a CSV row into the tick contract on_tick() expects.
        The CSV must carry array-valued columns as semicolon-separated
        strings, e.g. curve_prices = '75.10;75.35;75.62;75.80;76.01'."""
        def _arr(key: str, dtype=float) -> np.ndarray:
            raw = row[key]
            if isinstance(raw, (list, tuple, np.ndarray)):
                return np.asarray(raw, dtype=dtype)
            return np.asarray([dtype(x) for x in str(raw).split(";")])

        def _tup(key: str) -> tuple:
            raw = row[key]
            if isinstance(raw, (list, tuple)):
                return tuple(str(x) for x in raw)
            return tuple(str(raw).split(";"))

        return {
            "timestamp": row["timestamp"],
            "f1_price": float(row["f1_price"]),
            "f1_vol": int(row["f1_vol"]),
            "f1_symbol": str(row["f1_symbol"]),
            "f1_expiry": row["f1_expiry"],
            "f2_expiry": row["f2_expiry"],
            "curve_prices": _arr("curve_prices", float),
            "curve_spreads": _arr("curve_spreads", float),
            "curve_symbols": _tup("curve_symbols"),
            "l3_features": _arr("l3_features", float),
            "risk_free_rate": float(row["risk_free_rate"]),
        }
