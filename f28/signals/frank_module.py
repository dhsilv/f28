"""
Frank Engine (Phase 1): VPIN + KL-Divergence death-signal detector.

Notes on design choices that intentionally avoid common textbook traps:

* VPIN tick test: We use the Lee-Ready *sign-inheritance* rule for unchanged-
  tick prices, NOT the 50/50 split. The spec is explicit about this -- the
  split rule artificially deflates the order-flow imbalance on quiet ticks
  ("dilution") and is a known bias.

* KL-Divergence: The baseline distribution P is the OFFLINE steady-state
  distribution loaded from training data. The live distribution Q is the
  rolling window. Comparing live-vs-live (a window split against itself)
  is approximately zero by construction and useless as a regime detector.

* Both KDEs are evaluated on a shared support that covers the union of P and
  Q. This is required for a finite, well-defined KL.

* We do NOT assume Gaussian returns anywhere -- KDE is non-parametric. The
  Gaussian *kernel* is just a smoother; the resulting density is empirical.

* Death signal requires VPIN AND KL above threshold (per spec). An OR gate
  would fire on garden-variety volume bursts.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from scipy.stats import gaussian_kde, entropy

from signals.base_signal import BaseSignal


class FrankSignalEngine(BaseSignal):
    def __init__(
        self,
        vpin_threshold: float = 0.75,
        entropy_limit: float = 2.0,
        bucket_volume: int = 1000,
        window_size: int = 50,
        live_returns_window: int = 500,
        kl_recompute_every: int = 25,
        kde_bw: str | float = "silverman",
    ):
        # Hyperparameters
        self.vpin_threshold = float(vpin_threshold)
        self.entropy_limit = float(entropy_limit)
        self.bucket_volume = int(bucket_volume)
        self.window_size = int(window_size)
        self.live_returns_window = int(live_returns_window)
        self.kl_recompute_every = int(kl_recompute_every)
        self.kde_bw = kde_bw

        # VPIN state
        self.current_bucket_vol = 0
        self.current_buy_vol = 0.0
        self.current_sell_vol = 0.0
        self.bucket_imbalances: deque[float] = deque(maxlen=self.window_size)
        self.last_price: Optional[float] = None
        self.last_tick_sign: int = 0  # for inherit-on-unchanged

        # Entropy state
        self.live_returns: deque[float] = deque(maxlen=self.live_returns_window)
        self._last_price_for_returns: Optional[float] = None

        # Offline-trained baseline (set via load_baseline)
        self._baseline_returns: Optional[np.ndarray] = None
        self._baseline_kde: Optional[gaussian_kde] = None

        # KL caching: avoid refitting KDE on every tick
        self._cached_kl: Optional[float] = None
        self._ticks_since_kl: int = 0

    # ------------------------------------------------------------------
    # Offline parameter ingestion
    # ------------------------------------------------------------------
    def load_baseline(self, baseline_returns: np.ndarray) -> None:
        """Ingest the offline-trained steady-state log-return sample.

        We store the raw sample (not just an evaluated PDF) so that the
        live KDE and baseline KDE can share a dynamic grid that always
        covers the union of both supports.
        """
        arr = np.asarray(baseline_returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 50:
            raise ValueError("Baseline returns sample is too small for a stable KDE.")
        self._baseline_returns = arr
        self._baseline_kde = gaussian_kde(arr, bw_method=self.kde_bw)

    # ------------------------------------------------------------------
    # Tick ingestion
    # ------------------------------------------------------------------
    def process_tick(self, timestamp, price: float, volume: int) -> None:
        self._update_vpin_state(price, volume)
        self._update_entropy_state(price)
        self._ticks_since_kl += 1

    def is_death_signal_triggered(self) -> bool:
        vpin = self._calculate_vpin()
        kl = self._kl_divergence_cached()
        if vpin is None or kl is None:
            return False
        # Spec: BOTH must breach. Single-axis fires are not death signals.
        return (vpin > self.vpin_threshold) and (kl > self.entropy_limit)

    def get_signal_name(self) -> str:
        return "Frank_VPIN_KDE_v2"

    # ------------------------------------------------------------------
    # VPIN
    # ------------------------------------------------------------------
    def _update_vpin_state(self, price: float, volume: int) -> None:
        if self.last_price is None:
            self.last_price = price
            return

        # Lee-Ready with sign INHERITANCE on zero ticks (spec-compliant)
        if price > self.last_price:
            sign = +1
        elif price < self.last_price:
            sign = -1
        else:
            sign = self.last_tick_sign  # may be 0 until first directional move

        if sign > 0:
            self.current_buy_vol += volume
        elif sign < 0:
            self.current_sell_vol += volume
        # sign == 0 (cold start, no direction yet): drop volume from the
        # imbalance accounting rather than dilute it.

        self.last_tick_sign = sign if sign != 0 else self.last_tick_sign
        self.current_bucket_vol += volume
        self.last_price = price

        if self.current_bucket_vol >= self.bucket_volume:
            imbalance = abs(self.current_buy_vol - self.current_sell_vol)
            self.bucket_imbalances.append(imbalance)
            self.current_bucket_vol = 0
            self.current_buy_vol = 0.0
            self.current_sell_vol = 0.0

    def _calculate_vpin(self) -> Optional[float]:
        if len(self.bucket_imbalances) < self.window_size:
            return None
        total_imbalance = float(sum(self.bucket_imbalances))
        total_volume = self.window_size * self.bucket_volume
        return total_imbalance / total_volume

    # ------------------------------------------------------------------
    # KL-Divergence: baseline (P, offline) || live (Q, rolling)
    # ------------------------------------------------------------------
    def _update_entropy_state(self, price: float) -> None:
        if self._last_price_for_returns is None:
            self._last_price_for_returns = price
            return
        if price <= 0 or self._last_price_for_returns <= 0:
            return
        r = np.log(price / self._last_price_for_returns)
        self._last_price_for_returns = price
        if np.isfinite(r):
            self.live_returns.append(float(r))

    def _kl_divergence_cached(self) -> Optional[float]:
        # Refit KDE only every N ticks. KDE refit is O(N) and per-tick
        # cost would dominate the OnQuote loop.
        if self._ticks_since_kl < self.kl_recompute_every and self._cached_kl is not None:
            return self._cached_kl

        kl = self._compute_kl_divergence()
        if kl is not None:
            self._cached_kl = kl
            self._ticks_since_kl = 0
        return self._cached_kl

    def _compute_kl_divergence(self) -> Optional[float]:
        if self._baseline_kde is None or self._baseline_returns is None:
            # No offline baseline -> cannot compute. Return None so the
            # AND gate in is_death_signal_triggered will hold its fire.
            return None
        if len(self.live_returns) < max(50, self.live_returns_window // 4):
            return None

        live = np.fromiter(self.live_returns, dtype=float)
        try:
            kde_q = gaussian_kde(live, bw_method=self.kde_bw)
        except (np.linalg.LinAlgError, ValueError):
            # Singular covariance -- typically a flat-price burn-in window
            return None

        # Shared grid spanning the UNION of supports. Required for KL to
        # be well-defined; otherwise we can clip the tail mass that drives
        # the divergence.
        lo = float(min(self._baseline_returns.min(), live.min()))
        hi = float(max(self._baseline_returns.max(), live.max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        # Pad slightly so KDE tails are properly captured.
        pad = 0.1 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, 1024)

        p = self._baseline_kde.evaluate(grid)
        q = kde_q.evaluate(grid)

        # Add a numerical floor uniformly so neither distribution can have
        # a hard zero in the support of the other -> KL stays finite.
        eps = 1e-12
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)

        # scipy.stats.entropy normalizes p and q internally to sum to 1,
        # so the absolute scale of the grid spacing drops out. The result
        # is the discrete-grid approximation of D_KL(P || Q).
        return float(entropy(pk=p, qk=q))
