"""
Almgren-Chriss optimal execution trajectory with HMM-driven urgency.

x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)

where X is the initial inventory, T is the total number of slices, and
kappa is the risk-aversion / urgency parameter derived from the current
HMM regime.

Design corrections vs. v1:

1. Uses the O(1) Forward Algorithm (`predict_online_fast`) on the HMM,
   per spec. v1 used the Viterbi decode, which defeats the whole point
   of the fast forward scheme.

2. The fractional-lot accumulator is flushed at T. The old code dropped
   any remainder on the final step, meaning the roll could finish one
   contract short. The entire purpose of the accumulator is exact
   discrete execution; v1 violated that purpose on the last tick.

3. The abs-value on `actual_trade = int(np.floor(total_target))` could
   go NEGATIVE if kappa regime flips mid-trajectory and the sinh curve
   non-monotonically "wants" to add back. We clip at zero -- we never
   un-trade. Inventory monotonicity is a property of a proper sinh
   trajectory with fixed kappa, but kappa can change between ticks, so
   we have to enforce monotonicity explicitly.

4. `emergency_liquidate` actually dumps the remaining inventory, rather
   than just flipping a flag.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


class ExecutionEngine:
    def __init__(
        self,
        hmm_model,
        total_time_steps: int = 10,
        kappa_map: Optional[dict[int, float]] = None,
    ):
        self.hmm = hmm_model
        self.T = int(total_time_steps)

        # 0 = Quiet (low urgency, ~TWAP), 1 = Trending, 2 = Distressed
        self.kappa_map = kappa_map if kappa_map is not None else {
            0: 0.1,
            1: 1.5,
            2: 5.0,
        }

        self.is_executing = False
        self.target_tenor: Optional[str] = None
        self.initial_qty: int = 0
        self.remaining_qty: int = 0
        self.current_step: int = 0
        self.fractional_remainder: float = 0.0

    # ------------------------------------------------------------------
    def initiate_roll(self, target_tenor: str, target_qty: int) -> None:
        self.is_executing = True
        self.target_tenor = target_tenor
        self.initial_qty = int(target_qty)
        self.remaining_qty = int(target_qty)
        self.current_step = 0
        self.fractional_remainder = 0.0

    def get_next_order_size(self, current_l3_features: np.ndarray) -> int:
        """Tick-by-tick desired child order size. Returns integer lots."""
        if not self.is_executing:
            return 0
        if self.remaining_qty <= 0:
            self.is_executing = False
            return 0

        # Final slice: flush everything left, accumulator included.
        if self.current_step >= self.T - 1:
            final = self.remaining_qty
            self.remaining_qty = 0
            self.fractional_remainder = 0.0
            self.current_step += 1
            self.is_executing = False
            return max(0, int(final))

        # O(1) forward algorithm -> regime -> kappa
        regime = self.hmm.predict_online_fast(current_l3_features)
        kappa = self.kappa_map.get(int(regime), 1.0)

        inv_t = self._sinh_inventory(self.current_step, kappa)
        inv_next = self._sinh_inventory(self.current_step + 1, kappa)
        theoretical_trade = inv_t - inv_next

        total_target = theoretical_trade + self.fractional_remainder

        # Floor to integer contracts; never negative (no un-trading)
        actual_trade = max(0, int(math.floor(total_target)))

        # Can't trade more than we have left
        actual_trade = min(actual_trade, self.remaining_qty)

        self.fractional_remainder = total_target - actual_trade
        self.remaining_qty -= actual_trade
        self.current_step += 1

        return actual_trade

    # ------------------------------------------------------------------
    def _sinh_inventory(self, t: int, kappa: float) -> float:
        """Remaining inventory at slice t under kappa."""
        if t >= self.T:
            return 0.0
        if kappa < 1e-4:
            # TWAP limit: linear decay
            return self.initial_qty * (1.0 - t / self.T)

        # sinh(kappa * T) can overflow for large kappa*T. Guard by using
        # the exp-form identity: sinh(a)/sinh(b) = (exp(a-b) - exp(-a-b)) /
        # (1 - exp(-2b))
        a = kappa * (self.T - t)
        b = kappa * self.T
        if b > 50:  # overflow regime, use the identity
            num = math.exp(a - b) - math.exp(-a - b)
            den = 1.0 - math.exp(-2.0 * b)
            if den <= 0:
                return 0.0
            return self.initial_qty * (num / den)

        return self.initial_qty * (math.sinh(a) / math.sinh(b))

    # ------------------------------------------------------------------
    def emergency_liquidate(self) -> int:
        """Halt the sinh trajectory and flush remaining inventory as a
        market order. Returns the quantity the caller must immediately
        route."""
        qty = int(self.remaining_qty)
        self.is_executing = False
        self.remaining_qty = 0
        self.fractional_remainder = 0.0
        return qty
