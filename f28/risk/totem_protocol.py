"""
Totem Protocol: Phase 4 parallel-running circuit breaker.

Two independent tests against the underlying continuous-diffusion
assumption. Either test failing trips the breaker:

1. RV vs BPV. Realized Variance captures everything (diffusion + jumps).
   Bipower Variation isolates the diffusion piece because adjacent
   returns cannot both be large in a pure diffusion. A large RV/BPV
   ratio is evidence of a structural jump.

       RV  = sum_i r_i^2
       BPV = (pi/2) * (n / (n-1)) * sum_{i>=2} |r_i| * |r_{i-1}|

   The n/(n-1) correction matters for small windows (Barndorff-Nielsen
   & Shephard 2004). The old code omitted it.

2. Hurst exponent by variance scaling:
       Var( r^(k) ) = k^(2H) * Var( r^(1) )
   where r^(k) is the k-aggregated return. H < 0.15 indicates the
   market is deterministically pinned (extreme anti-persistence,
   typical on delivery days when a delivery algo is forcing price).

Data semantics: the series being monitored is supplied externally. The
master feeds in F1 tick log-returns directly. Monitoring the basis
(F1 - Spot) is only sensible if spot is actually provided and is truly
spot, not "F1 itself." See strategy/f28_master.py.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np


class TotemCircuitBreaker:
    def __init__(
        self,
        window_size: int = 100,
        jump_threshold: float = 2.5,
        hurst_limit: float = 0.15,
        min_samples: int = 30,
    ):
        self.window_size = int(window_size)
        self.jump_threshold = float(jump_threshold)
        self.hurst_limit = float(hurst_limit)
        self.min_samples = int(min_samples)

        self.price_history: deque[float] = deque(maxlen=self.window_size + 1)
        self.log_returns: deque[float] = deque(maxlen=self.window_size)

        self.is_halted: bool = False
        self.last_halt_reason: Optional[str] = None

    # ------------------------------------------------------------------
    def is_market_broken(self, current_price: float, timestamp=None) -> bool:
        """Feed one price tick. Returns True iff the system should halt."""
        if self.is_halted:
            return True

        if current_price <= 0 or not np.isfinite(current_price):
            return False

        self.price_history.append(float(current_price))
        if len(self.price_history) >= 2:
            p_prev = self.price_history[-2]
            p_curr = self.price_history[-1]
            if p_prev > 0 and p_curr > 0:
                r = math.log(p_curr / p_prev)
                if np.isfinite(r):
                    self.log_returns.append(r)

        if len(self.log_returns) < self.min_samples:
            return False

        # 1. Jump test
        rv, bpv = self._rv_bpv()
        if bpv > 0 and (rv / bpv) > self.jump_threshold:
            self._trip(f"RV/BPV={rv/bpv:.2f} > {self.jump_threshold}", timestamp)
            return True

        # 2. Hurst pinning test
        h = self._hurst_variance_scaling()
        if h is not None and h < self.hurst_limit:
            self._trip(f"H={h:.3f} < {self.hurst_limit}", timestamp)
            return True

        return False

    # ------------------------------------------------------------------
    def _rv_bpv(self) -> tuple[float, float]:
        r = np.fromiter(self.log_returns, dtype=float)
        n = r.size
        rv = float(np.sum(r * r))
        if n < 2:
            return rv, 0.0
        abs_r = np.abs(r)
        bpv_raw = float(np.sum(abs_r[1:] * abs_r[:-1]))
        # BN-S small-sample correction
        bpv = (math.pi / 2.0) * (n / (n - 1)) * bpv_raw
        return rv, bpv

    def _hurst_variance_scaling(self) -> Optional[float]:
        r = np.fromiter(self.log_returns, dtype=float)
        if r.size < max(20, self.min_samples):
            return None

        var_1 = float(np.var(r, ddof=1))
        if var_1 <= 0:
            return 0.0  # Flat price: infinite pinning

        # Non-overlapping 2-aggregation is a cleaner estimator than the
        # overlapping one the old code used.
        n_pairs = r.size // 2
        r2 = r[: 2 * n_pairs].reshape(n_pairs, 2).sum(axis=1)
        if r2.size < 10:
            return None
        var_2 = float(np.var(r2, ddof=1))
        if var_2 <= 0:
            return 0.0

        ratio = var_2 / var_1
        if ratio <= 0:
            return 0.0

        h = 0.5 * math.log2(ratio)
        # Clamp to the valid support [0, 1]. Values outside are pure
        # finite-sample noise.
        return max(0.0, min(1.0, h))

    # ------------------------------------------------------------------
    def _trip(self, reason: str, timestamp) -> None:
        self.is_halted = True
        self.last_halt_reason = reason
        print(f"[{timestamp}] TOTEM PROTOCOL HALT: {reason}")
        print(">>> EMERGENCY LIQUIDATION REQUIRED. <<<")
        print(">>> DO NOT RETRAIN ON PRE-BREAK DATA. <<<")

    def manual_reset(self) -> None:
        """Spec: the breaker requires manual reset and a regime burn-in
        before the strategy can resume. This is the reset entry point."""
        self.is_halted = False
        self.last_halt_reason = None
        self.price_history.clear()
        self.log_returns.clear()
