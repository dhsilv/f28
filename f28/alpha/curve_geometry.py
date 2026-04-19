"""
PCA Model on the EWMA-decayed forward-curve covariance.

Bug fixes vs. v1:

1. The Roll Gap Stitch needs to know which physical contract is currently
   in the F1 slot. The old code referenced an undefined `current_f1_symbol`
   inside `_update_ewma_covariance`. We now thread the symbol through every
   call explicitly. The stitch must operate on the slot whose contract just
   changed -- typically F1, but conceptually any slot.

2. `_calculate_residuals` USED `self.last_prices` AFTER the covariance
   update had already overwritten it with `current_prices`. The log-return
   was therefore identically zero and the strategy could never see a rich
   or cheap tenor. Residuals now operate on the same return vector that
   the covariance update consumed, computed once.

3. Initialization is via a true burn-in counter, not "set the EWMA mean
   to whatever the first observation happened to be."

4. Sign convention is documented and verified: a POSITIVE residual means
   the realized return was richer than the PCA reconstruction of the
   first k principal components. To roll into something CHEAP we want
   the most NEGATIVE residual (after spread costs).
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


class PCAModel:
    def __init__(
        self,
        num_tenors: int = 5,
        ewma_span: int = 60,
        num_components: int = 3,
        liquidity_penalty_bps: float = 2.0,
        f3_penalty_multiplier: float = 1.5,
        burn_in: int = 30,
    ):
        self.num_tenors = int(num_tenors)
        self.num_components = int(num_components)
        self.liquidity_penalty = float(liquidity_penalty_bps) / 10000.0
        self.f3_penalty_multiplier = float(f3_penalty_multiplier)

        # EWMA decay: pandas-style span -> alpha
        # alpha = 2 / (span + 1); decay (lambda) = 1 - alpha
        alpha = 2.0 / (ewma_span + 1.0)
        self.decay = 1.0 - alpha

        self.ewma_mean = np.zeros(self.num_tenors)
        self.ewma_cov = np.eye(self.num_tenors)
        self.eigenvectors = np.eye(self.num_tenors)

        self.last_prices: Optional[np.ndarray] = None
        # Symbol per slot, e.g. ['ESZ24', 'ESH25', 'ESM25', 'ESU25', 'ESZ25']
        self.last_symbols: Optional[Tuple[str, ...]] = None

        self.burn_in = int(burn_in)
        self._n_observed = 0

    # ------------------------------------------------------------------
    def _update(
        self,
        current_prices: np.ndarray,
        current_symbols: Tuple[str, ...],
    ) -> Optional[np.ndarray]:
        """Advance EWMA and return the realized centered return vector
        (or None if this tick was a stitch / cold start)."""
        current_prices = np.asarray(current_prices, dtype=float)

        if self.last_prices is None:
            self.last_prices = current_prices.copy()
            self.last_symbols = tuple(current_symbols)
            return None

        # ---------- THE STITCH ----------
        # Per-slot symbol comparison. If ANY slot rolled, the cross-slot
        # log-return for that slot is meaningless. We zero out the affected
        # slot's return so the EWMA covariance is not corrupted, but we
        # still allow the unaffected slots to contribute -- otherwise a
        # single F1 expiry day kills the entire covariance update.
        prev_symbols = self.last_symbols or tuple(current_symbols)
        rolled_mask = np.array(
            [a != b for a, b in zip(prev_symbols, current_symbols)],
            dtype=bool,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            raw_returns = np.log(current_prices / self.last_prices)
        # Replace stitched slots with the EWMA mean (i.e. zero centered
        # contribution after subtracting the mean below).
        if rolled_mask.any():
            raw_returns[rolled_mask] = self.ewma_mean[rolled_mask]

        # Persist new state immediately
        self.last_prices = current_prices.copy()
        self.last_symbols = tuple(current_symbols)

        if self._n_observed < self.burn_in:
            # Accumulate but do not yet trust the EWMA / PCA outputs.
            # Use a simple running mean during burn-in.
            self._n_observed += 1
            self.ewma_mean = (
                self.ewma_mean * (self._n_observed - 1) + raw_returns
            ) / self._n_observed
            return None

        # 1. Update EWMA mean
        self.ewma_mean = self.decay * self.ewma_mean + (1.0 - self.decay) * raw_returns
        centered = raw_returns - self.ewma_mean

        # 2. Update EWMA covariance: Sigma_t = lambda*Sigma_{t-1} + (1-lambda)*r r^T
        outer = np.outer(centered, centered)
        self.ewma_cov = self.decay * self.ewma_cov + (1.0 - self.decay) * outer

        return centered

    def _perform_pca(self) -> None:
        # eigh: symmetric PSD, ascending order. Reverse for descending.
        eigvals, eigvecs = np.linalg.eigh(self.ewma_cov)
        idx = np.argsort(eigvals)[::-1]
        self.eigenvectors = eigvecs[:, idx]

    def _residuals_from_centered(self, centered: np.ndarray) -> np.ndarray:
        P_k = self.eigenvectors[:, : self.num_components]
        factor_scores = P_k.T @ centered
        reconstruction = P_k @ factor_scores
        # epsilon = realized - reconstructed
        # > 0 means tenor priced rich vs. its PCA-implied move -> sell
        # < 0 means tenor priced cheap                        -> buy
        return centered - reconstruction

    # ------------------------------------------------------------------
    def get_optimal_roll_target(
        self,
        tick_data: dict,
        current_f1_qty: int,
    ) -> Tuple[str, int]:
        """Return (target_tenor, target_qty).

        tick_data must contain:
          'prices'  : np.ndarray of length num_tenors  -- ordered [F1..Fn]
          'spreads' : np.ndarray of length num_tenors  -- bid-ask widths
          'symbols' : tuple[str] of length num_tenors  -- physical contract IDs
        """
        prices = np.asarray(tick_data["prices"], dtype=float)
        spreads = np.asarray(tick_data["spreads"], dtype=float)
        symbols = tuple(tick_data["symbols"])

        centered = self._update(prices, symbols)

        # During burn-in, default to F2 (the boring sequential roll).
        if centered is None or self._n_observed < self.burn_in:
            return "F2", current_f1_qty

        self._perform_pca()
        residuals = self._residuals_from_centered(centered)

        # We are BUYING the back month. Cheap = negative epsilon = good.
        # "edge" is positive when buying below model.
        raw_edge_f2 = -residuals[1]
        raw_edge_f3 = -residuals[2]

        # Liquidity adjustment: half-spread cost in return units, plus a
        # constant penalty for crossing the further-out (less liquid) book.
        cost_f2 = 0.5 * spreads[1] / prices[1] + self.liquidity_penalty
        cost_f3 = 0.5 * spreads[2] / prices[2] + self.liquidity_penalty * self.f3_penalty_multiplier

        edge_f2 = raw_edge_f2 - cost_f2
        edge_f3 = raw_edge_f3 - cost_f3

        # Skip-roll only if F3 strictly beats F2 net of costs AND has
        # positive expected edge in absolute terms.
        if edge_f3 > edge_f2 and edge_f3 > 0:
            return "F3", current_f1_qty
        return "F2", current_f1_qty
