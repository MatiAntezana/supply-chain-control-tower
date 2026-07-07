"""LightGBM quantile forecaster: one booster per quantile, direct multi-horizon.

Because every autoregressive feature has lag >= horizon (28d), the same model
predicts any day in the next 28 without recursion and without leakage.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_contract

MODELS_DIR = REPO_ROOT / "models"


def feature_columns(contract: dict) -> tuple[list[str], list[str]]:
    cats = contract["categoricals"]
    feats = list(contract["features"].keys()) + cats
    return feats, cats


class QuantileForecaster:
    def __init__(self, quantiles: list[float], lgbm_params: dict, seed: int):
        self.quantiles = quantiles
        self.lgbm_params = lgbm_params
        self.seed = seed
        self.boosters: dict[float, lgb.Booster] = {}
        self.contract = load_contract()

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        feats, cats = feature_columns(self.contract)
        x = df[feats].copy()
        for c in cats:
            x[c] = x[c].astype("category")
        return x

    def fit(self, train: pd.DataFrame) -> QuantileForecaster:
        x = self._prep(train)
        y = train["units"].astype(float)
        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                random_state=self.seed,
                deterministic=True,
                force_row_wise=True,
                verbose=-1,
                **self.lgbm_params,
            )
            model.fit(x, y)
            self.boosters[q] = model.booster_
        return self

    def predict(self, df: pd.DataFrame) -> dict[float, np.ndarray]:
        x = self._prep(df)
        preds = {}
        prev = None
        for q in sorted(self.quantiles):
            p = np.clip(self.boosters[q].predict(x), 0, None)
            if prev is not None:
                p = np.maximum(p, prev)  # enforce non-crossing quantiles
            preds[q] = p
            prev = p
        return preds

    def save(self, tag: str) -> list[Path]:
        MODELS_DIR.mkdir(exist_ok=True)
        paths = []
        for q, booster in self.boosters.items():
            path = MODELS_DIR / f"lgbm_{tag}_p{int(q * 100)}.txt"
            booster.save_model(str(path))
            paths.append(path)
        return paths

    @classmethod
    def load(cls, quantiles: list[float], tag: str) -> QuantileForecaster:
        obj = cls(quantiles, {}, 0)
        for q in quantiles:
            path = MODELS_DIR / f"lgbm_{tag}_p{int(q * 100)}.txt"
            obj.boosters[q] = lgb.Booster(model_file=str(path))
        return obj
