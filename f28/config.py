"""
Config loader. Reads f28/config.json and returns a plain dict.

Why a JSON file and not a Python module: the C++ Strategy Studio port
will parse the same file so the live Python and live C++ deployments
run on identical hyperparameters by construction. A .py config would
fork the moment anyone edited it.

Keys that start with "_" are treated as documentation-only and stripped
on load so modules never see them.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("f28.config")

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent / "config.json")


def _strip_comments(obj: Any) -> Any:
    """Recursively drop keys starting with '_' (used for inline docs)."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(x) for x in obj]
    return obj


def load_config(path: str | None = None) -> dict:
    """Load f28 hyperparameters. Raises FileNotFoundError if missing --
    the strategy is not meant to run on module defaults silently."""
    path = path or os.environ.get("F28_CONFIG", DEFAULT_CONFIG_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"F28 config not found at {path}. Set F28_CONFIG env var or "
            "pass path explicitly."
        )
    with open(path, "r") as f:
        raw = json.load(f)
    cfg = _strip_comments(raw)

    # Normalize integer-keyed dicts that had to be stringified in JSON.
    if "execution" in cfg and "kappa_map" in cfg["execution"]:
        cfg["execution"]["kappa_map"] = {
            int(k): float(v) for k, v in cfg["execution"]["kappa_map"].items()
        }

    logger.info("Loaded F28 config from %s (commodity=%s)", path, cfg.get("commodity"))
    return cfg
