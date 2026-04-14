"""
Microstructure HMM for LOB regime classification.

Per spec:

* Three logical states: Quiet (0), Trending (1), Distressed (2).
* Baum-Welch training is OFFLINE.
* Live prediction is the O(1) Forward Algorithm, NOT a Viterbi decode
  over a rolling window. We kept the Viterbi path for offline evaluation
  only and made the fast Forward path the default for the event loop.

Performance fixes:

* Pre-invert each state's covariance matrix once on load/train. The hot
  path no longer calls scipy.stats.multivariate_normal, which is ~two
  orders of magnitude slower than a hand-rolled log-density.
* We propagate in LOG space to avoid underflow at extreme anomalies,
  then re-normalize.

Ingestion:

* `load_from_params(dict)` consumes the JSON exported by
  ops/train_f28_models.py so the live strategy is not forced to retrain.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence

import numpy as np


class MicrostructureHMM:
    def __init__(self, n_states: int = 3):
        self.n_states = int(n_states)

        # Parameters, populated by fit_offline() or load_from_params()
        self.transmat_: Optional[np.ndarray] = None      # (K, K)
        self.means_: Optional[np.ndarray] = None         # (K, D)
        self.covars_: Optional[np.ndarray] = None        # (K, D, D)
        self.startprob_: Optional[np.ndarray] = None     # (K,)

        # Cached for hot path: log-det and inverse of each cov
        self._inv_covars: Optional[np.ndarray] = None    # (K, D, D)
        self._log_norm: Optional[np.ndarray] = None      # (K,)

        self.state_map: dict[int, int] = {}
        self.is_trained: bool = False

        # Online forward state
        self._log_probs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit_offline(self, historical_l3_features: np.ndarray) -> None:
        """Run Baum-Welch via hmmlearn, then cache hot-path structures."""
        try:
            from hmmlearn.hmm import GaussianHMM  # lazy import
        except ImportError as exc:
            raise ImportError(
                "hmmlearn is required for fit_offline(). "
                "For live runs, use load_from_params() instead."
            ) from exc

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=1000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(np.asarray(historical_l3_features, dtype=float))

        self.transmat_ = np.asarray(model.transmat_, dtype=float)
        self.means_ = np.asarray(model.means_, dtype=float)
        self.covars_ = np.asarray(model.covars_, dtype=float)
        self.startprob_ = np.asarray(model.startprob_, dtype=float)

        self._finalize_load()

    def load_from_params(self, params: dict) -> None:
        """Hydrate from the JSON produced by train_f28_models.py."""
        self.transmat_ = np.asarray(params["transition_matrix"], dtype=float)
        self.means_ = np.asarray(params["means"], dtype=float)
        self.covars_ = np.asarray(params["covars"], dtype=float)
        self.startprob_ = np.asarray(params["start_prob"], dtype=float)

        k = self.means_.shape[0]
        if k != self.n_states:
            self.n_states = k
        self._finalize_load()

    def _finalize_load(self) -> None:
        self._map_hidden_states()
        self._precompute_emission_cache()
        self._log_probs = np.log(np.clip(self.startprob_, 1e-300, None))
        self.is_trained = True

    def _map_hidden_states(self) -> None:
        """Order states by total covariance trace: smallest trace = Quiet,
        largest trace = Distressed. Works for any n_states."""
        traces = np.array([np.trace(c) for c in self.covars_])
        sorted_idx = np.argsort(traces)  # ascending
        # Logical labels: 0, 1, ..., n_states-1 correspond to ascending
        # trace (i.e. quieter -> more distressed). For n_states=3 this
        # matches the spec's Quiet=0, Trending=1, Distressed=2.
        self.state_map = {
            int(internal_idx): logical_idx
            for logical_idx, internal_idx in enumerate(sorted_idx)
        }

    def _precompute_emission_cache(self) -> None:
        K, D, _ = self.covars_.shape
        self._inv_covars = np.empty_like(self.covars_)
        self._log_norm = np.empty(K)
        for i in range(K):
            cov = self.covars_[i]
            # Regularize in case of near-singular cov from short training data
            cov_reg = cov + np.eye(D) * 1e-10
            sign, logdet = np.linalg.slogdet(cov_reg)
            if sign <= 0:
                raise np.linalg.LinAlgError(f"State {i} covariance is not PD.")
            self._inv_covars[i] = np.linalg.inv(cov_reg)
            self._log_norm[i] = -0.5 * (D * math.log(2.0 * math.pi) + logdet)

    # ------------------------------------------------------------------
    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        """Return log p(x | state_i) for each i. Hot path."""
        K = self.n_states
        out = np.empty(K)
        for i in range(K):
            diff = x - self.means_[i]
            quad = diff @ self._inv_covars[i] @ diff
            out[i] = self._log_norm[i] - 0.5 * quad
        return out

    def predict_online_fast(self, current_features: Sequence[float]) -> int:
        """O(1) Forward update. Returns the logical state id."""
        if not self.is_trained:
            raise RuntimeError("HMM must be trained or loaded before prediction.")

        x = np.asarray(current_features, dtype=float)

        log_em = self._log_emission(x)

        # log(prev @ transmat) = logsumexp over prior state index
        # Using numpy-only logsumexp to avoid a scipy import on the hot path.
        log_T = np.log(np.clip(self.transmat_, 1e-300, None))
        # shape ops: prev (K,), log_T (K, K): result_j = logsumexp_i(prev_i + log_T[i, j])
        combined = self._log_probs[:, None] + log_T  # (K, K)
        m = np.max(combined, axis=0)
        log_prior = m + np.log(np.sum(np.exp(combined - m[None, :]), axis=0))

        new_log_probs = log_em + log_prior

        # Normalize in log space
        m_total = np.max(new_log_probs)
        total = m_total + math.log(float(np.sum(np.exp(new_log_probs - m_total))))
        self._log_probs = new_log_probs - total

        internal = int(np.argmax(self._log_probs))
        return self.state_map[internal]

    def reset_online(self) -> None:
        """Reset forward probabilities to the stationary / startprob state.
        Called by the master when exiting HALTED after a burn-in."""
        if self.startprob_ is None:
            return
        self._log_probs = np.log(np.clip(self.startprob_, 1e-300, None))

    # ------------------------------------------------------------------
    def predict_batch_viterbi(self, obs_matrix: np.ndarray) -> np.ndarray:
        """OFFLINE ONLY: Viterbi decode for evaluation / research notebooks.
        Do NOT call from the live event loop."""
        if not self.is_trained:
            raise RuntimeError("HMM must be trained or loaded before prediction.")
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(n_components=self.n_states, covariance_type="full")
        model.startprob_ = self.startprob_
        model.transmat_ = self.transmat_
        model.means_ = self.means_
        model.covars_ = self.covars_
        _, hidden = model.decode(np.asarray(obs_matrix, dtype=float))
        return np.array([self.state_map[int(h)] for h in hidden])
