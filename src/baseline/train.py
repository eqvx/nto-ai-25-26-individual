"""
Main training script for the LightGBM model.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

from . import config, constants


def train() -> None:
    """Runs the model training pipeline.

    Loads prepared data from data/processed/, then trains a LightGBM model
    for each of the 5 folds using GroupKFold cross-validation. Trained models
    are saved to the directory specified in the config.

    Note: Data must be prepared first using prepare_data.py
    """
    # Load prepared data
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    # Separate train and test sets
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Define features (X) and target (y)
    features = [
        col for col in train_set.columns if col not in [constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION]
    ]
    # Exclude any remaining object columns that are not model features
    non_feature_object_cols = train_set[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X = train_set[features]
    y = train_set[config.TARGET]
    groups = train_set[constants.COL_USER_ID]

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Cross-validation training
    gkf = GroupKFold(n_splits=config.N_SPLITS)

    print(f"Starting training with {config.N_SPLITS}-fold GroupKFold CV...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"--- Fold {fold + 1}/{config.N_SPLITS} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Initialize and train the model
        model = lgb.LGBMRegressor(**config.LGB_PARAMS)

        # Update fit params with early stopping callback for the current fold
        fit_params = config.LGB_FIT_PARAMS.copy()
        fit_params["callbacks"] = [lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False)]

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=fit_params["eval_metric"],
            callbacks=fit_params["callbacks"],
        )

        # Evaluate the model
        val_preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mae = mean_absolute_error(y_val, val_preds)
        print(f"Fold {fold + 1} Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # Save the trained model
        model_path = config.MODEL_DIR / config.MODEL_FILENAME_PATTERN.format(fold=fold)
        model.booster_.save_model(str(model_path))
        print(f"Model for fold {fold + 1} saved to {model_path}")

    print("Training complete.")


if __name__ == "__main__":
    train()
