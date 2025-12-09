from pathlib import Path

# Paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default files
DEFAULT_TRAIN_PATH = RAW_DIR / "application_train.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "model_lgbm.joblib"
DEFAULT_PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
DEFAULT_FEATURE_LIST_PATH = MODELS_DIR / "feature_list.json"

# Target & columns
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# Validation
RANDOM_STATE = 42
N_SPLITS = 5  # StratifiedKFold
EVAL_METRIC = "roc_auc"

# Model params (can be overridden via CLI or optuna)
LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "n_estimators": 400,
    "num_leaves": 64,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}