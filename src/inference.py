import json
from pathlib import Path
from typing import List, Union

import joblib
import pandas as pd

import src.config as config


class Predictor:
    def __init__(
        self,
        model_path: Path = config.DEFAULT_MODEL_PATH,
        feature_list_path: Path = config.DEFAULT_FEATURE_LIST_PATH,
    ):
        self.model_path = model_path
        self.feature_list_path = feature_list_path
        self.model = joblib.load(model_path)
        with open(feature_list_path) as f:
            self.feature_cols = json.load(f)

    def predict_proba(self, data: Union[dict, list]) -> List[float]:
        df = self._to_df(data)
        
        # Check for missing required features
        missing_features = set(self.feature_cols) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features: {sorted(missing_features)}. "
                f"Please provide all required features."
            )
        
        # Align columns to match training feature order
        df = df[self.feature_cols]
        probs = self.model.predict_proba(df)[:, 1].tolist()
        return probs

    def predict(self, data: Union[dict, list], threshold: float = 0.5):
        probs = self.predict_proba(data)
        preds = [int(p >= threshold) for p in probs]
        return preds, probs

    def _to_df(self, data: Union[dict, list]) -> pd.DataFrame:
        if isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError("data must be dict or list of dicts")
