"""
F28 entry point for CL (WTI Crude Oil) futures.

Wires every injected dependency and boots the backtester. Python mirror
of the OnQuote master class's construction block in the C++ Strategy
Studio deployment -- owns the object graph, nothing else.

Underlying: CL (NYMEX WTI Crude Oil). Rate source is short-term
(SOFR 3M / 3M T-Bill), NOT DGS10, because the front-back basis has
tau ~ 1 month.

To run:
    1. ops/train_f28_models.py -> ./models/f28_parameters.json
    2. python -m f28.main

Placeholders (OU calibration) are flagged inline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from alpha.curve_geometry import PCAModel
from alpha.ekf_overlay import ConvenienceYieldEKF
from engine.backtester import TickEngine
from execution.almgren_chriss import ExecutionEngine
from execution.hmm_regime import MicrostructureHMM
from risk.totem_protocol import TotemCircuitBreaker
from signals.frank_module import FrankSignalEngine
from strategy.f28_master import F28Strategy


PARAMS_PATH = "./models/f28_parameters.json"
DATA_PATH = "./data/f1_f2_f3_ticks.csv"


def _load_trained_params(path: str) -> dict | None:
    """Return offline-trained parameters, or None if unavailable."""
    if not os.path.exists(path):
        print(
            f"WARNING: {path} not found. Running with uninitialized Frank "
            "baseline and an untrained HMM -- strategy will NOT fire. "
            "Run ops/train_f28_models.py first."
        )
        return None
    with open(path, "r") as f:
        return json.load(f)


def build_strategy(params: dict | None) -> F28Strategy:
    # -------- Phase 1: Frank --------
    # CL tuning: bucket_volume ~1000 matches typical CL tick volume per
    # bucket; ES would need 10-50x larger buckets.
    frank = FrankSignalEngine(
        vpin_threshold=0.75,
        entropy_limit=2.0,
        bucket_volume=1000,
        window_size=50,
        live_returns_window=500,
    )
    if params is not None and "frank_kde" in params:
        import numpy as np
        baseline = np.asarray(params["frank_kde"]["returns_sample"], dtype=float)
        frank.load_baseline(baseline)

    # -------- Phase 2: PCA (5 tenors, 3 PCs) --------
    pca = PCAModel(
        num_tenors=5,
        ewma_span=60,
        num_components=3,
        liquidity_penalty_bps=2.0,
        burn_in=30,
    )

    # -------- Phase 2.5: EKF (CL / WTI calibration) --------
    # kappa, theta, sigma_y should be offline-fit against a historical
    # convenience-yield series derived from the F2/F1 basis. The defaults
    # below are reasonable starting values for WTI; replace with MLE fits
    # before live deployment.
    #   theta = 0.03  --  3% annualized long-run convenience yield
    #   sigma_y = 0.25 --  CL convenience yield is genuinely volatile
    #   kappa = 1.5   --  moderate mean reversion
    #   physical_limit = 0.15 -- 15% y ==> real supply shock (Cushing,
    #                            OPEC, geopolitical). Drives the Phase 2
    #                            PCA override.
    #   obs_noise = 0.10 -- F2 microstructure noise in $ terms
    ekf = ConvenienceYieldEKF(
        kappa=1.5,
        theta=0.03,
        sigma_y=0.25,
        obs_noise=0.10,
        physical_limit=0.15,
    )

    # -------- Phase 3: HMM + Almgren-Chriss --------
    hmm = MicrostructureHMM(n_states=3)
    if params is not None and "hmm_matrices" in params:
        hmm.load_from_params(params["hmm_matrices"])
    execution = ExecutionEngine(hmm_model=hmm, total_time_steps=20)

    # -------- Phase 4: Totem --------
    # price_floor is CL-specific. The April 2020 WTI event proved prices
    # can go negative; $5 is well above any plausible normal-regime price
    # and well below any plausible panic low, so it catches the regime
    # break without triggering on ordinary selloffs.
    totem = TotemCircuitBreaker(
        window_size=100,
        jump_threshold=2.5,
        hurst_limit=0.15,
        price_floor=5.0,
    )

    return F28Strategy(
        frank_engine=frank,
        pca_engine=pca,
        ekf_overlay=ekf,
        hmm_engine=hmm,
        execution_engine=execution,
        totem_protocol=totem,
        initial_f1_qty=1000,
    )


def main() -> None:
    params = _load_trained_params(PARAMS_PATH)
    strategy = build_strategy(params)

    engine = TickEngine(data_path=DATA_PATH)
    engine.attach_strategy(strategy)

    print("Initiating Project F-28...")
    if Path(DATA_PATH).exists():
        engine.run()
    else:
        print(
            f"No data file at {DATA_PATH}. The TickEngine is wired; pass a "
            "stream to engine.run_stream(iterable_of_ticks) from your "
            "external parser to begin replay."
        )


if __name__ == "__main__":
    main()
