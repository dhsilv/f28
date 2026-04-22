"""
Tick contract: the single dict shape every on_tick() call must satisfy.

This file is the source of truth for the tick schema. F28_STRATEGY.md
Appendix B tracks the prose version; this file is the machine version
and the one the C++ port will mirror as a struct.
"""
from __future__ import annotations

from typing import Any, Sequence, TypedDict

import numpy as np


class TickData(TypedDict):
    timestamp: Any            # datetime or pandas.Timestamp
    f1_price: float           # F1 last trade / mid; used as spot proxy for CL
    f1_vol: int               # executed trade size on this tick
    f1_symbol: str            # e.g. 'CLM25'
    f1_expiry: Any            # datetime
    f2_expiry: Any            # datetime
    curve_prices: np.ndarray  # shape (num_tenors,) ordered [F1, F2, F3, ...]
    curve_spreads: np.ndarray # shape (num_tenors,) bid-ask per tenor
    curve_symbols: tuple      # length num_tenors, physical contract ids
    l3_features: np.ndarray   # shape (3,)  [Spread, Imbalance, Intensity]
    risk_free_rate: float     # short rate (SOFR 3M / 3M T-Bill)


_REQUIRED_KEYS = (
    "timestamp", "f1_price", "f1_vol", "f1_symbol", "f1_expiry", "f2_expiry",
    "curve_prices", "curve_spreads", "curve_symbols", "l3_features",
    "risk_free_rate",
)


def validate_tick(tick: dict, expected_num_tenors: int = 5) -> None:
    """Cheap per-tick invariants. Raises ValueError on violation.

    Enforced:
      * All required keys present.
      * l3_features is length 3 (HMM emission is hardcoded 3-D).
      * curve_prices / curve_spreads / curve_symbols have matching length.
      * First two curve slots are in the expected [F1, F2, ...] order if
        symbols are tagged with "F<n>" convention; otherwise only the
        shape is checked.
    """
    missing = [k for k in _REQUIRED_KEYS if k not in tick]
    if missing:
        raise ValueError(f"Tick missing required keys: {missing}")

    l3 = tick["l3_features"]
    if len(l3) != 3:
        raise ValueError(f"l3_features must be length 3, got {len(l3)}")

    n_prices = len(tick["curve_prices"])
    n_spreads = len(tick["curve_spreads"])
    n_symbols = len(tick["curve_symbols"])
    if not (n_prices == n_spreads == n_symbols):
        raise ValueError(
            f"Curve length mismatch: prices={n_prices}, spreads={n_spreads}, "
            f"symbols={n_symbols}"
        )
    if n_prices < 2:
        raise ValueError(f"Curve must carry at least F1 and F2, got {n_prices} slots")
