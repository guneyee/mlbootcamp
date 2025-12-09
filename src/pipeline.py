import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import src.config as config


def load_data(train_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    return df


def build_preprocessor(df: pd.DataFrame):
    target = config.TARGET_COL
    id_col = config.ID_COL
    feature_cols = [c for c in df.columns if c not in [target, id_col]]

    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if df[c].dtype != "object"]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor, feature_cols, cat_cols, num_cols


def train_and_eval(df: pd.DataFrame, preprocessor, feature_cols):
    X = df[feature_cols]
    y = df[config.TARGET_COL]

    model = LGBMClassifier(**config.LGBM_PARAMS)
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    cv = StratifiedKFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE
    )
    oof_pred = cross_val_predict(
        clf, X, y, cv=cv, method="predict_proba", n_jobs=-1, verbose=0
    )[:, 1]
    auc = roc_auc_score(y, oof_pred)

    # Fit on full data
    clf.fit(X, y)
    return clf, auc


def save_artifacts(clf, feature_cols, preprocessor_path, model_path, feature_list_path):
    # Pipeline already contains preprocessor+model; save pipeline as model artifact
    joblib.dump(clf, model_path)
    with open(feature_list_path, "w") as f:
        json.dump(feature_cols, f, indent=2)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=config.DEFAULT_TRAIN_PATH)
    parser.add_argument("--model-path", type=Path, default=config.DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--preprocessor-path", type=Path, default=config.DEFAULT_PREPROCESSOR_PATH
    )
    parser.add_argument(
        "--feature-list-path", type=Path, default=config.DEFAULT_FEATURE_LIST_PATH
    )
    parsed = parser.parse_args(args)

    df = load_data(parsed.train_path)
    preprocessor, feature_cols, cat_cols, num_cols = build_preprocessor(df)
    clf, auc = train_and_eval(df, preprocessor, feature_cols)

    save_artifacts(
        clf, feature_cols, parsed.preprocessor_path, parsed.model_path, parsed.feature_list_path
    )

    print(f"OOF AUC: {auc:.4f}")
    print(f"Saved model to {parsed.model_path}")
    print(f"Saved feature list to {parsed.feature_list_path}")


if __name__ == "__main__":
    main()
