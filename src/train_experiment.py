"""
Automated Experiment Orchestrator
- Features: calls feature modules to build all features
- Models: trains LightGBM and neural recommender, logs to MLflow
- Ready for HPO (Optuna), parallel or batch runs

This script provides a reproducible, centralized entry point for all ML experiments.
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Feature modules
from src.features.tfidf_features import TfidfFeatureGenerator
from src.features.bert_features import compute_bert_embeddings
from src.features.user_book_embeddings import UserBookEmbedder
# Model modules
from src.models.neural_recommender import NeuralRecommender, train_nn
from src.models.ensemble import ModelEnsemble

# Optional: MLflow
try:
    import mlflow
    MLFLOW_ON = True
except ImportError:
    print("[INFO] MLflow not installed, tracking disabled.")
    MLFLOW_ON = False

# Optional: LightGBM
try:
    import lightgbm as lgb
    LGBM_ON = True
except ImportError:
    print("[WARN] LightGBM not installed, skipping LightGBM training.")
    LGBM_ON = False

def main(cfg):
    # ------------
    # Feature engineering
    # ------------
    print("[Stage] TF-IDF Features...")
    tfg = TfidfFeatureGenerator(
        max_features=cfg['tfidf_max_features'],
        min_df=cfg['tfidf_min_df'],
        max_df=cfg['tfidf_max_df'],
        ngram_range=tuple(cfg['tfidf_ngram_range'])
    )
    tfg.fit(cfg['data_train'], text_col=cfg['desc_col'])
    tfidf_train = tfg.transform(cfg['data_train'], text_col=cfg['desc_col'])
    tfidf_val = tfg.transform(cfg['data_val'], text_col=cfg['desc_col'])

    print("[Stage] BERT Embeddings...")
    bert_train = compute_bert_embeddings(cfg['data_train'][[cfg['book_id_col'], cfg['desc_col']]],
                                         model_name=cfg['bert_model'],
                                         description_col=cfg['desc_col'],
                                         id_col=cfg['book_id_col'],
                                         batch_size=cfg['bert_batch_size'],
                                         max_length=cfg['bert_max_length'])
    bert_val = compute_bert_embeddings(cfg['data_val'][[cfg['book_id_col'], cfg['desc_col']]],
                                       model_name=cfg['bert_model'],
                                       description_col=cfg['desc_col'],
                                       id_col=cfg['book_id_col'],
                                       batch_size=cfg['bert_batch_size'],
                                       max_length=cfg['bert_max_length'])

    print("[Stage] User/Book Embeddings...")
    embedder = UserBookEmbedder(no_components=cfg['lfm_dim'])
    embedder.fit(cfg['interactions_train'], user_col=cfg['user_id_col'],
                 book_col=cfg['book_id_col'], rating_col=cfg['rating_col'])
    lfm_user_train, lfm_book_train = embedder.get_embedding_dfs()
    # Join everything — TODO: Customize merge keys & protection
    # X_train = ...  concatenate all features
    # X_val = ...

    # ------------
    # Train LightGBM (if available)
    # ------------
    if LGBM_ON:
        print("[Stage] LightGBM training...")
        lgbm = lgb.LGBMRegressor(**cfg['lgb_params'])
        lgbm.fit(cfg['X_train_lgb'], cfg['y_train'],
                 eval_set=[(cfg['X_val_lgb'], cfg['y_val'])],
                 early_stopping_rounds=cfg['early_stopping_rounds'],
                 eval_metric="rmse", verbose=10)
        lgbm_preds = lgbm.predict(cfg['X_val_lgb'])
    else:
        lgbm_preds = np.zeros(len(cfg['y_val']))

    # ------------
    # Train Neural Recommender
    # ------------
    print("[Stage] Neural model training...")
    # Prepare categorical and dense arrays
    # X_cat = [user_id_arr, book_id_arr, author_arr, ...]
    # X_dense = numpy array (dense features, e.g., BERT + TFIDF)
    # train_nn(...)
    nn_preds = np.zeros(len(cfg['y_val']))  # TODO: Fill with real runs

    # ------------
    # Ensemble
    # ------------
    print("[Stage] Ensemble...")
    ens = ModelEnsemble()
    if LGBM_ON: ens.add_model("lgbm", lambda X: lgbm.predict(X))
    ens.add_model("nn", lambda X: nn_preds)
    all_preds = {
        "lgbm": lgbm_preds,
        "nn": nn_preds
    }
    stack_pred = ens.predict(None, model_preds=all_preds, method="mean")
    rmse = np.sqrt(np.mean((cfg['y_val'] - stack_pred) ** 2))
    mae = np.mean(np.abs(cfg['y_val'] - stack_pred))

    print(f"[RESULTS] Ensemble RMSE={rmse:.4f}, MAE={mae:.4f}")
    if MLFLOW_ON:
        mlflow.start_run(run_name=cfg.get('exp_name','exp'))
        mlflow.log_params(cfg)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.end_run()

    print("[ALL DONE]")

if __name__ == "__main__":
    # TODO: support config files/CLI/YAML integration
    parser = argparse.ArgumentParser()
    # Add args as needed, or read config from path
    # parser.add_argument('...')
    args = parser.parse_args()
    # for demo: fill cfg with stubs
    dummy = np.random.RandomState(0)
    N = 100
    cfg = dict(
        tfidf_max_features=500, tfidf_min_df=2, tfidf_max_df=0.95, tfidf_ngram_range=[1,2],
        desc_col="description", book_id_col="book_id", user_id_col="user_id", author_id_col="author_id",
        bert_model="bert-base-multilingual-cased", bert_batch_size=8, bert_max_length=128,
        lfm_dim=32, rating_col="rating",
        early_stopping_rounds=10,
        lgb_params={"n_estimators":32, "learning_rate":0.1, "num_leaves":8},
        X_train_lgb=dummy.normal(0,1,(N,10)),
        X_val_lgb=dummy.normal(0,1,(N,10)),
        y_train=dummy.normal(5,2,N),
        y_val=dummy.normal(5,2,N),
        data_train=pd.DataFrame({
            "description": ["foo bar bla"]*N, "book_id": [str(i%10) for i in range(N)]
        }),
        data_val=pd.DataFrame({
            "description": ["foo bar bla"]*N, "book_id": [str(i%10) for i in range(N)]
        }),
        interactions_train=pd.DataFrame({
            "user_id": [str(i%20) for i in range(N)],
            "book_id": [str(i%10) for i in range(N)],
            "rating": dummy.randint(1,6,N)
        }),
        exp_name="stub-exp"
    )
    main(cfg)
