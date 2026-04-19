"""
Extended Kalman Filter over the Gibson-Schwartz one-factor pricing model.

State: y_t (convenience yield), assumed to follow an OU process
    dy = kappa * (theta - y) dt + sigma_y dW

Observation: F_t = S_t * exp((r_t - y_t) * tau_t) + noise

Design notes:

* The spec calls for dynamic tau and r. We expose them as per-tick inputs
  and do NOT hardcode anything in the filter. Whoever calls step() is
  responsible for computing tau from (expiry_date - now) and pulling r
  from DGS10. See strategy/f28_master.py for the actual wiring.

* The exact OU transition variance over dt is
        V_dt = (sigma_y^2 / (2*kappa)) * (1 - exp(-2*kappa*dt))
  For small kappa*dt this collapses to sigma_y^2 * dt. We use the exact
  form so the filter stays consistent if dt varies (asynchronous ticks)
  or if kappa*dt is not small.

* The EKF linearizes h(y) = S*exp((r-y)*tau) around the prior mean, which
  implicitly Gaussianizes the innovation. This IS a Gaussian assumption,
  unavoidable for a Kalman-family filter. We flag jump conditions to the
  Totem Protocol rather than trying to handle them inside the EKF.

* Supply-shock condition: the spec says "if y EXCEEDS a critical threshold"
  in either direction. A high positive y means steep backwardation
  (scarcity). A very negative y means extreme contango (glut). Either
  invalidates the smooth PCA residual story. We flag on |y| > limit.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from constants import SECONDS_PER_YEAR

logger = logging.getLogger(__name__)

# 1-second cadence in calendar-time years. Matches f28_master._compute_dt.
_DEFAULT_DT_ONE_SECOND = 1.0 / SECONDS_PER_YEAR


class ConvenienceYieldEKF:
    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma_y: float,
        obs_noise: float,
        physical_limit: float = 0.15,
        default_dt: float = _DEFAULT_DT_ONE_SECOND,  # ~1-second tick, calendar-time
        initial_uncertainty: float = 0.01,
    ):
        """
        Parameters
        ----------
        kappa, theta, sigma_y : OU parameters for y
        obs_noise             : std. dev. of microstructure noise (price units)
        physical_limit        : |y| above this = supply-side regime break
        default_dt             : dt used if caller does not supply one
        initial_uncertainty    : prior variance on y at boot
        """
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_y = float(sigma_y)
        self.obs_noise = float(obs_noise)
        self.physical_limit = float(physical_limit)
        self.default_dt = float(default_dt)

        # Scalar state: variance is a scalar, kept as 1x1 for consistency.
        self.y: float = float(theta)
        self.P: float = float(initial_uncertainty)

        self.R: float = self.obs_noise ** 2  # observation noise variance

        # Store prior for reset()
        self._initial_uncertainty: float = float(initial_uncertainty)

    # ------------------------------------------------------------------
    def reset(self, keep_y: bool = True) -> None:
        """Reset filter covariance, optionally keep the point estimate.

        Called by the master on events that invalidate the prior
        uncertainty (Totem halt + resume, or a contract roll where the
        tau discontinuity makes P stale). Default is to keep `y` because
        convenience yield is a market state, not a per-contract state --
        the mean estimate is still a reasonable starting point post-roll.
        """
        if not keep_y:
            self.y = float(self.theta)
        self.P = self._initial_uncertainty

    # ------------------------------------------------------------------
    def _process_noise(self, dt: float) -> float:
        """Exact OU transition variance over dt."""
        two_kdt = 2.0 * self.kappa * dt
        if two_kdt < 1e-10:
            return self.sigma_y ** 2 * dt
        return (self.sigma_y ** 2 / (2.0 * self.kappa)) * (1.0 - math.exp(-two_kdt))

    def step(
        self,
        S_t: float,
        F_market: float,
        tau: float,
        r: float,
        dt: Optional[float] = None,
    ) -> bool:
        """One EKF cycle. Returns True if a supply-side regime break is
        detected (|y| > physical_limit after update)."""
        if dt is None:
            dt = self.default_dt

        self._predict(dt)
        self._update(S_t, F_market, tau, r)

        return abs(self.y) > self.physical_limit

    # ------------------------------------------------------------------
    def _predict(self, dt: float) -> None:
        # Exact OU mean reversion: y_{t+dt} = theta + (y_t - theta) * exp(-kappa*dt)
        decay = math.exp(-self.kappa * dt)
        self.y = self.theta + (self.y - self.theta) * decay
        # Jacobian of the exact transition: dy_new/dy_old = decay
        self.P = decay * self.P * decay + self._process_noise(dt)

    def _update(self, S_t: float, F_market: float, tau: float, r: float) -> None:
        # Pre-validate inputs before any arithmetic. A degenerate input
        # (expired contract, NaN from feed corruption, pathological tau)
        # would otherwise poison F_theo and self.y silently.
        if not (math.isfinite(S_t) and math.isfinite(F_market)
                and math.isfinite(tau) and math.isfinite(r)):
            logger.warning("EKF update skipped: non-finite input "
                           "(S_t=%s, F=%s, tau=%s, r=%s)", S_t, F_market, tau, r)
            return
        if tau <= 0 or S_t <= 0:
            return  # observation model is undefined at/past expiry

        # Theoretical observation
        F_theo = S_t * math.exp((r - self.y) * tau)

        # Jacobian H = d(F_theo)/d(y) = -tau * F_theo
        H = -tau * F_theo

        if not (math.isfinite(F_theo) and math.isfinite(H)):
            logger.warning("EKF update skipped: F_theo or H overflowed")
            return

        innovation = F_market - F_theo
        S_cov = H * self.P * H + self.R

        # Belt-and-suspenders guard on the innovation covariance.
        if S_cov <= 0 or not np.isfinite(S_cov):
            return

        K = (self.P * H) / S_cov

        self.y = self.y + K * innovation
        # Joseph form is more numerically stable than (I - KH) P for scalar,
        # but scalar case is trivially stable. Keep the simple form.
        self.P = (1.0 - K * H) * self.P

    def get_current_yield(self) -> float:
        return float(self.y)

    def get_uncertainty(self) -> float:
        return float(self.P)
