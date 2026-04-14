"""
F28 master FSM. The spec: almost zero math, strict dependency injection.

Orchestration responsibilities:

* Maintain the FSM state (HOLDING_F1, ROLLING, COMPLETED, HALTED).
* Feed the parallel Totem circuit breaker on every tick.
* Compute tau DYNAMICALLY from the F1 expiry date on every tick.
* Pull r from the supplied DGS10 curve (no hardcoded 5%).
* Pass the current F1 contract symbol to the PCA so the Roll Gap Stitch
  actually has the information it needs to fire.

Tick contract (tick_data dict keys):

    timestamp       : datetime or pandas.Timestamp
    spot_price      : float                       -- true spot (index / cash)
    f1_price        : float
    f1_vol          : int                         -- executed trade size on this tick
    f1_symbol       : str                         -- e.g. 'CLM25'
    f1_expiry       : datetime                    -- physical expiry
    curve_prices    : np.ndarray, shape (n,)      -- [F1, F2, F3, ...]
    curve_spreads   : np.ndarray, shape (n,)      -- bid-ask per tenor
    curve_symbols   : tuple[str,  ...]            -- symbol per slot
    l3_features     : np.ndarray                  -- [Spread, Imbalance, Intensity]
    risk_free_rate  : float                       -- DGS10, continuously compounded
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class F28Strategy:
    STATES = ("HOLDING_F1", "ROLLING", "COMPLETED", "HALTED")

    def __init__(
        self,
        frank_engine,
        pca_engine,
        ekf_overlay,
        hmm_engine,
        execution_engine,
        totem_protocol,
        initial_f1_qty: int,
    ):
        self.frank = frank_engine
        self.pca = pca_engine
        self.ekf = ekf_overlay
        self.hmm = hmm_engine
        self.execution = execution_engine
        self.totem = totem_protocol

        self.state: str = "HOLDING_F1"
        self.f1_inventory: int = int(initial_f1_qty)

        # Tick-delta bookkeeping for EKF dt
        self._last_tick_ts = None

    # ------------------------------------------------------------------
    def on_tick(self, tick_data: dict) -> None:
        ts = tick_data["timestamp"]

        # -------- Phase 4: parallel circuit breaker (always first) --------
        # Feed F1 price directly -- log-returns of F1 are what the BPV /
        # Hurst tests are validated against. Basis vs. spot is noisier
        # and conflates two instruments.
        if self.totem.is_market_broken(tick_data["f1_price"], ts):
            if self.state != "HALTED":
                self._execute_emergency_halt(tick_data)
            return

        # -------- FSM routing --------
        if self.state == "HOLDING_F1":
            self._evaluate_roll_initiation(tick_data)
        elif self.state == "ROLLING":
            self._execute_roll_trajectory(tick_data)
        # COMPLETED / HALTED: absorb ticks silently

        self._last_tick_ts = ts

    # ------------------------------------------------------------------
    def _evaluate_roll_initiation(self, tick_data: dict) -> None:
        # 1. Feed Frank
        self.frank.process_tick(
            timestamp=tick_data["timestamp"],
            price=tick_data["f1_price"],
            volume=tick_data["f1_vol"],
        )

        if not self.frank.is_death_signal_triggered():
            return

        ts = tick_data["timestamp"]
        print(f"[{ts}] F-28: Frank triggered death signal on {tick_data['f1_symbol']}.")

        # 2. Phase 2.5 -- EKF physics overlay
        tau = self._compute_tau(tick_data)
        if tau <= 0:
            # Past expiry. Skip EKF, force F2 roll.
            is_supply_shock = True
        else:
            dt = self._compute_dt(ts)
            is_supply_shock = self.ekf.step(
                S_t=tick_data["spot_price"],
                F_market=tick_data["f1_price"],
                tau=tau,
                r=tick_data["risk_free_rate"],
                dt=dt,
            )

        # 3. Target selection
        if is_supply_shock:
            print(
                f"[{ts}] F-28: EKF physical override (y={self.ekf.get_current_yield():.4f}). "
                "Bypassing PCA, rolling to F2."
            )
            target_tenor, target_qty = "F2", self.f1_inventory
        else:
            pca_tick = {
                "prices": tick_data["curve_prices"],
                "spreads": tick_data["curve_spreads"],
                "symbols": tick_data["curve_symbols"],
            }
            target_tenor, target_qty = self.pca.get_optimal_roll_target(
                pca_tick, self.f1_inventory
            )

        self.execution.initiate_roll(target_tenor, target_qty)
        self.state = "ROLLING"

    # ------------------------------------------------------------------
    def _execute_roll_trajectory(self, tick_data: dict) -> None:
        trade_size = self.execution.get_next_order_size(tick_data["l3_features"])

        if trade_size > 0:
            self._send_order(
                venue="PRIMARY_EXCHANGE",
                tenor=self.execution.target_tenor,
                side="BUY",
                qty=trade_size,
                price=self._quote_price_for_target(tick_data),
            )
            # Offset side: selling F1. In the real book we send both legs
            # as a spread order, but the mock keeps it one-legged.
            self._send_order(
                venue="PRIMARY_EXCHANGE",
                tenor="F1",
                side="SELL",
                qty=trade_size,
                price=tick_data["f1_price"],
            )
            self.f1_inventory -= trade_size

            if self.f1_inventory <= 0:
                print(f"[{tick_data['timestamp']}] F-28: Roll complete.")
                self.state = "COMPLETED"

    # ------------------------------------------------------------------
    def _execute_emergency_halt(self, tick_data: dict) -> None:
        self.state = "HALTED"
        qty_to_dump = self.execution.emergency_liquidate()
        print(
            f"[{tick_data['timestamp']}] F-28 MASTER: HALTED. "
            f"Market liquidating {qty_to_dump} contracts of "
            f"{self.execution.target_tenor}."
        )
        if qty_to_dump > 0:
            self._send_order(
                venue="PRIMARY_EXCHANGE",
                tenor=self.execution.target_tenor or "F1",
                side="BUY",
                qty=qty_to_dump,
                price=tick_data["f1_price"],
                order_type="MARKET",
            )

    # ------------------------------------------------------------------
    def _compute_tau(self, tick_data: dict) -> float:
        """Time-to-maturity in years, continuously compounded convention."""
        expiry = tick_data["f1_expiry"]
        now = tick_data["timestamp"]
        delta = expiry - now
        seconds = delta.total_seconds()
        # 252 * 6.5 * 3600 active seconds per year is one convention; for
        # continuously traded commodity futures calendar years are cleaner.
        return seconds / (365.25 * 24.0 * 3600.0)

    def _compute_dt(self, ts) -> float:
        if self._last_tick_ts is None:
            return self.ekf.default_dt
        secs = (ts - self._last_tick_ts).total_seconds()
        if secs <= 0:
            return self.ekf.default_dt
        return secs / (365.25 * 24.0 * 3600.0)

    def _quote_price_for_target(self, tick_data: dict) -> float:
        """Look up the target tenor's current price from curve_prices."""
        idx = {"F1": 0, "F2": 1, "F3": 2, "F4": 3, "F5": 4}
        i = idx.get(self.execution.target_tenor, 1)
        return float(tick_data["curve_prices"][i])

    # ------------------------------------------------------------------
    def _send_order(
        self,
        venue: str,
        tenor: str,
        side: str,
        qty: int,
        price: float,
        order_type: str = "LIMIT",
    ) -> None:
        print(
            f"[ORDER OUT] {order_type} {side} {qty} {tenor} @ {price:.4f} | "
            f"Venue: {venue}"
        )
