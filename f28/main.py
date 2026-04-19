"""
F28 entry point for CL (WTI Crude Oil) futures.

Wires every injected dependency and boots the backtester. Python mirror
of the OnQuote master class's construction block in the C++ Strategy
Studio deployment -- owns the object graph, nothing else.

Underlying: CL (NYMEX WTI Crude Oil). Rate source is short-term
(SOFR 3M / 3M T-Bill), NOT DGS10, because the front-back basis has
tau ~ 1 month.

All hyperparameters are read from config.json (single source of truth;
the C++ port reads the same file). Override the path via the F28_CONFIG
env var if needed.

To run:
    1. ops/train_f28_models.py -> ./models/f28_parameters.json
    2. python -m f28.main
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from alpha.curve_geometry import PCAModel
from alpha.ekf_overlay import ConvenienceYieldEKF
from config import load_config
from engine.backtester import TickEngine
from execution.almgren_chriss import ExecutionEngine
from execution.hmm_regime import MicrostructureHMM
from risk.totem_protocol import TotemCircuitBreaker
from signals.frank_module import FrankSignalEngine
from strategy.f28_master import F28Strategy


PARAMS_PATH = "./models/f28_parameters.json"
DATA_PATH = "./data/f1_f2_f3_ticks.csv"

logger = logging.getLogger("f28.main")


def _load_trained_params(path: str) -> dict | None:
    """Return offline-trained parameters, or None if unavailable."""
    if not os.path.exists(path):
        logger.warning(
            "%s not found. Running with uninitialized Frank baseline and "
            "an untrained HMM -- strategy will NOT fire. Run "
            "ops/train_f28_models.py first.",
            path,
        )
        return None
    with open(path, "r") as f:
        return json.load(f)


def build_strategy(params: dict | None, cfg: dict) -> F28Strategy:
    # -------- Phase 1: Frank --------
    frank = FrankSignalEngine(**cfg["frank"])
    if params is not None and "frank_kde" in params:
        baseline = np.asarray(params["frank_kde"]["returns_sample"], dtype=float)
        frank.load_baseline(baseline)
    else:
        logger.warning(
            "Frank baseline not provided -- KL divergence will return None "
            "and the death signal cannot fire."
        )

    # -------- Phase 2: PCA (5 tenors, 3 PCs) --------
    pca = PCAModel(**cfg["pca"])

    # -------- Phase 2.5: EKF (CL / WTI calibration) --------
    # kappa, theta, sigma_y should be offline-fit against a historical
    # convenience-yield series derived from the F2/F1 basis. The config
    # values are reasonable starting values for WTI; replace with MLE
    # fits before live deployment.
    ekf = ConvenienceYieldEKF(**cfg["ekf"])

    # -------- Phase 3: HMM + Almgren-Chriss --------
    hmm = MicrostructureHMM(n_states=cfg["hmm"]["n_states"])
    if params is not None and "hmm_matrices" in params:
        hmm.load_from_params(params["hmm_matrices"])
    else:
        logger.warning(
            "HMM matrices not provided -- regime classification cannot run "
            "and ExecutionEngine will raise on first predict_online_fast()."
        )
    exec_cfg = cfg["execution"]
    execution = ExecutionEngine(
        hmm_model=hmm,
        total_time_steps=exec_cfg["total_time_steps"],
        kappa_map=exec_cfg["kappa_map"],
    )

    # -------- Phase 4: Totem --------
    totem = TotemCircuitBreaker(**cfg["totem"])

    return F28Strategy(
        frank_engine=frank,
        pca_engine=pca,
        ekf_overlay=ekf,
        hmm_engine=hmm,
        execution_engine=execution,
        totem_protocol=totem,
        initial_f1_qty=cfg["strategy"]["initial_f1_qty"],
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    params = _load_trained_params(PARAMS_PATH)
    strategy = build_strategy(params, cfg)

    engine = TickEngine(data_path=DATA_PATH)
    engine.attach_strategy(strategy)

    logger.info("Initiating Project F-28 (commodity=%s)...", cfg.get("commodity"))
    if Path(DATA_PATH).exists():
        engine.run()
    else:
        logger.info(
            "No data file at %s. The TickEngine is wired; pass a stream to "
            "engine.run_stream(iterable_of_ticks) from your external parser "
            "to begin replay.",
            DATA_PATH,
        )


if __name__ == "__main__":
    main()
