import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from src.models.neural_recommender import NeuralRecommender, train_nn
from src.models.ensemble import ModelEnsemble
from src.baseline import config, constants

def main():
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    assert processed_path.exists(), f"Processed data not found at {processed_path}"
    print(f"Loading processed data from {processed_path} ...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")

    # Split (reproduce train.py logic)
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    test_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()

    # Temporal split for validation
    from src.baseline.temporal_split import get_split_date_from_ratio, temporal_split_by_date
    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Temporal split at {split_date}")
    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    # Compute aggregates as in train.py
    from src.baseline.features import add_aggregate_features, handle_missing_values
    train_split_with_agg = add_aggregate_features(train_split.copy(), train_split)
    val_split_with_agg = add_aggregate_features(val_split.copy(), train_split)
    train_final = handle_missing_values(train_split_with_agg, train_split)
    val_final   = handle_missing_values(val_split_with_agg, train_split)

    # Feature selection
    exclude_cols = [constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, constants.COL_TIMESTAMP]
    features = [col for col in train_final.columns if col not in exclude_cols]
    non_feature_object_cols = train_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    # Data for models
    X_train = train_final[features]
    y_train = train_final[config.TARGET]
    X_val = val_final[features]
    y_val = val_final[config.TARGET]

    # 1. LightGBM
    print("\n[Model] Training LightGBM ...")
    lgbm = lgb.LGBMRegressor(**config.LGB_PARAMS)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)],
             eval_metric="rmse", callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)])
    val_preds_lgbm = lgbm.predict(X_val)
    print(f"LightGBM RMSE: {mean_squared_error(y_val, val_preds_lgbm, squared=False):.4f}")
    print(f"LightGBM MAE:  {mean_absolute_error(y_val, val_preds_lgbm):.4f}")

    # 2. NeuralRecommender
    print("\n[Model] Training NeuralRecommender ...")
    cat_cols = [col for col in config.CAT_FEATURES if col in features]  # Only use categoricals that exist in features
    if len(cat_cols) == 0:
        print("Warning: No categorical features found, skipping NeuralRecommender")
        val_preds_nn = np.zeros(len(y_val))
    else:
        X_train_cat = [X_train[col].astype('category').cat.codes.values for col in cat_cols]
        X_val_cat   = [X_val[col].astype('category').cat.codes.values for col in cat_cols]
        dense_cols = [f for f in features if f not in cat_cols]
        X_train_dense = X_train[dense_cols].values.astype('float32')
        X_val_dense   = X_val[dense_cols].values.astype('float32')
        cat_info = [(c, int(X_train[c].nunique())) for c in cat_cols]
        nn_model = NeuralRecommender(cat_feat_info=cat_info, dense_dim=X_train_dense.shape[1])
        train_nn(nn_model, X_train_cat, X_train_dense, y_train.values, X_val_cat, X_val_dense, y_val.values)
        val_preds_nn = nn_model.predict(X_val_cat, X_val_dense)
        print(f"NeuralNet RMSE: {mean_squared_error(y_val, val_preds_nn, squared=False):.4f}")
        print(f"NeuralNet MAE:  {mean_absolute_error(y_val, val_preds_nn):.4f}")

    # 3. Ensemble
    print("\n[Model] Ensembling ...")
    ens = ModelEnsemble()
    ens.add_model("lgbm", lambda X: val_preds_lgbm)
    ens.add_model("nn",   lambda X: val_preds_nn)
    final_val_preds = ens.predict(None, model_preds={"lgbm": val_preds_lgbm, "nn": val_preds_nn}, method='mean')
    print(f"Ensemble RMSE: {mean_squared_error(y_val, final_val_preds, squared=False):.4f}")
    print(f"Ensemble MAE:  {mean_absolute_error(y_val, final_val_preds):.4f}")
    # TODO: Add meta-model stacking
    print("\nDone.")

if __name__ == "__main__":
    main()
