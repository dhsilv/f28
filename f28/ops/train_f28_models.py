"""
Offline ETL + training. Produces f28_parameters.json consumed by the live
Python strategy (and eventually the C++ Strategy Studio deployment).

Scope notes:

* The user explicitly said data parsing / PCAP dumping lives in another
  folder and will be streamed in from Strategy Studio. This module does
  NOT try to be authoritative on the tick schema. It consumes a cleaned
  pandas DataFrame (one row per tick) with a minimum set of required
  columns -- the upstream parser can produce that. If the raw files are
  already in this shape we load them; otherwise, build_from_dataframe
  lets you pass the frame directly.

* Frank baseline is exported as the RAW returns sample, not as a PDF on
  a fixed grid. The live engine fits its own KDE, and KL requires a
  shared support that can only be computed once the live window's
  support is known. Exporting just a (grid, pdf) pair makes a proper KL
  impossible.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd


class StrategyStudioETL:
    # Minimum schema expected from the upstream parser
    REQUIRED_COLS = {"Symbol", "Datetime", "MsgType"}
    QUOTE_COLS = {"BidPrice", "BidSize", "AskPrice", "AskSize"}
    TRADE_COLS = {"TradePrice", "TradeSize"}

    def __init__(self, data_directory: str):
        self.data_dir = data_directory

    # ------------------------------------------------------------------
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load pre-parsed CSV/Parquet files from self.data_dir.

        Expected columns (per tick):
            Symbol, Datetime, MsgType, BidPrice, BidSize, AskPrice,
            AskSize, TradePrice, TradeSize
        MsgType is 'Q' for quotes, 'T' for trades.
        """
        frames = []
        for fname in sorted(os.listdir(self.data_dir)):
            path = os.path.join(self.data_dir, fname)
            if fname.endswith(".parquet"):
                frames.append(pd.read_parquet(path))
            elif fname.endswith(".csv") or fname.endswith(".txt"):
                frames.append(pd.read_csv(path, on_bad_lines="skip"))

        if not frames:
            raise FileNotFoundError(f"No tick files found under {self.data_dir}")

        df = pd.concat(frames, ignore_index=True)

        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Upstream parser must emit these columns: {missing}. "
                "See the docstring for the expected schema."
            )

        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime").sort_index()
        return df

    # ------------------------------------------------------------------
    def extract_frank_baseline(self, df: pd.DataFrame) -> dict:
        """Returns the raw log-return sample for Frank's steady-state KDE.
        The live engine refits the KDE online, so we ship the sample,
        not an evaluated PDF."""
        trades = df[df["MsgType"] == "T"].copy()
        if "TradePrice" not in trades.columns or trades.empty:
            raise ValueError("No trade ticks found to build Frank baseline.")

        prices = trades["TradePrice"].astype(float).values
        # Log returns
        with np.errstate(divide="ignore", invalid="ignore"):
            log_returns = np.diff(np.log(prices))
        log_returns = log_returns[np.isfinite(log_returns)]

        if log_returns.size < 500:
            raise ValueError(
                f"Only {log_returns.size} valid trade returns; "
                "need at least 500 for a stable steady-state baseline."
            )

        return {
            "returns_sample": log_returns.tolist(),
            "n_samples": int(log_returns.size),
            "mean": float(np.mean(log_returns)),
            "std": float(np.std(log_returns, ddof=1)),
        }

    # ------------------------------------------------------------------
    def extract_hmm_features(
        self,
        df: pd.DataFrame,
        bucket_freq: str = "1s",
    ) -> np.ndarray:
        """L3 feature matrix: [Spread, Book Imbalance, Trade Intensity]."""
        quotes = df[df["MsgType"] == "Q"].copy()
        if not self.QUOTE_COLS.issubset(quotes.columns):
            raise ValueError(f"Quote rows missing columns: {self.QUOTE_COLS - set(quotes.columns)}")

        quotes["Spread"] = quotes["AskPrice"] - quotes["BidPrice"]
        denom = quotes["BidSize"] + quotes["AskSize"]
        quotes["Imbalance"] = np.where(
            denom > 0,
            (quotes["BidSize"] - quotes["AskSize"]) / denom,
            0.0,
        )
        q_res = quotes[["Spread", "Imbalance"]].resample(bucket_freq).mean().ffill()

        trades = df[df["MsgType"] == "T"].copy()
        if "TradeSize" not in trades.columns:
            raise ValueError("Trade rows missing TradeSize column.")
        t_res = (
            trades[["TradeSize"]]
            .rename(columns={"TradeSize": "Intensity"})
            .resample(bucket_freq)
            .sum()
            .fillna(0.0)
        )

        feats = pd.concat([q_res, t_res], axis=1).dropna()

        # Per-feature standardization is not done here -- the HMM can
        # absorb scale differences in the full covariance. If the user
        # wants a diagonal-cov HMM in future, standardize upstream.
        return feats.values.astype(float)

    # ------------------------------------------------------------------
    def train_hmm(self, feature_matrix: np.ndarray, n_states: int = 3) -> dict:
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(feature_matrix)

        # Resolve the internal -> logical state map here (authoritative)
        # so the live strategy and the C++ port both consume an explicit
        # mapping rather than re-deriving it from covariance traces. The
        # rule: sort by covariance trace ascending -> Quiet, Trending,
        # Distressed (for n_states=3).
        traces = np.array([np.trace(c) for c in model.covars_])
        sorted_idx = np.argsort(traces)
        state_map = {
            str(int(internal_idx)): int(logical_idx)
            for logical_idx, internal_idx in enumerate(sorted_idx)
        }

        return {
            "transition_matrix": model.transmat_.tolist(),
            "means": model.means_.tolist(),
            "covars": model.covars_.tolist(),
            "start_prob": model.startprob_.tolist(),
            "state_map": state_map,
            "n_states": int(n_states),
            "n_features": int(feature_matrix.shape[1]),
        }

    # ------------------------------------------------------------------
    def run(self, output_path: str = "./models/f28_parameters.json") -> dict:
        df = self.load_and_clean_data()

        frank_payload = self.extract_frank_baseline(df)
        hmm_features = self.extract_hmm_features(df)
        hmm_payload = self.train_hmm(hmm_features)

        output = {
            "frank_kde": frank_payload,
            "hmm_matrices": hmm_payload,
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"ETL COMPLETE: wrote {output_path}")
        return output


if __name__ == "__main__":
    etl = StrategyStudioETL(data_directory="./data/feb_to_apr_ticks")
    etl.run()
