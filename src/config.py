"""Central config loader. Every module gets its parameters from here."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = REPO_ROOT / "configs" / "params.yaml"
CONTRACT_PATH = REPO_ROOT / "configs" / "feature_contract.yaml"


def load_params(path: Path | None = None) -> dict:
    """Load params.yaml as a plain dict."""
    with open(path or PARAMS_PATH) as f:
        return yaml.safe_load(f)


def load_contract(path: Path | None = None) -> dict:
    """Load the versioned feature contract."""
    with open(path or CONTRACT_PATH) as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> np.random.Generator:
    """Seed python/numpy and return a dedicated Generator for local use."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def mlflow_uri(params: dict) -> str:
    return f"sqlite:///{REPO_ROOT / params['mlflow']['db_file']}"


def raw_dir(params: dict) -> Path:
    p = Path(params["data"]["raw_dir"])
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def processed_dir(params: dict) -> Path:
    d = REPO_ROOT / params["data"]["processed_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d
