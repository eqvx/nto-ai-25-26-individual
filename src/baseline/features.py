"""
Feature engineering script.
"""

import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import config, constants

# 1. Imports for new feature modules
# Update imports for poetry/python -m compatibility
from src.features.tfidf_features import TfidfFeatureGenerator
from src.features.bert_features import compute_bert_embeddings
from src.features.user_book_embeddings import UserBookEmbedder


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds user, book, and author aggregate features.

    Uses the training data to compute mean ratings and interaction counts
    to prevent data leakage from the test set.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion of the data for calculations.

    Returns:
        pd.DataFrame: The DataFrame with new aggregate features.
    """
    print("Adding aggregate features...")

    # User-based aggregates
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
    ]

    # Book-based aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
    ]

    # Author-based aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    return df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds the count of genres for each book.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.

    Returns:
        pd.DataFrame: The DataFrame with the new 'book_genres_count' column.
    """
    print("Adding genre features...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_GENRES_COUNT,
    ]
    return df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")


# --- REPLACEMENT: add_text_features, add_bert_features, plus collaborative embeddings ---
def add_text_features_new(df, desc_col="description", fit_on=None, cfg=None):
    # New TF-IDF pipeline
    tfg = TfidfFeatureGenerator(max_features=cfg.get("tfidf_max_features", 500), min_df=cfg.get("tfidf_min_df", 2),
                               max_df=cfg.get("tfidf_max_df", 0.95), ngram_range=tuple(cfg.get("tfidf_ngram_range", (1,2))))
    if fit_on is not None:
        tfg.fit(fit_on, text_col=desc_col)
    else:
        tfg.fit(df, text_col=desc_col)
    tfidf_df = tfg.transform(df, text_col=desc_col)
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    return df

def add_bert_features_new(df, desc_df, desc_col="description", id_col="book_id", cfg=None):
    # Check if BERT should be skipped
    if cfg and cfg.get("skip_bert", False):
        print("Skipping BERT features (skip_bert=True in config)")
        bert_dim = 768
        for i in range(bert_dim):
            df[f"bert_{i}"] = 0.0
        return df
    
    try:
        bert_df = compute_bert_embeddings(desc_df[[id_col, desc_col]],
                                          model_name=cfg.get("bert_model", "bert-base-multilingual-cased"),
                                          description_col=desc_col, id_col=id_col,
                                          batch_size=cfg.get("bert_batch_size", 2), 
                                          max_length=cfg.get("bert_max_length", 128))
        return df.merge(bert_df, on=id_col, how="left")
    except (RuntimeError, MemoryError, OSError, Exception) as e:
        print(f"Warning: BERT feature generation failed: {e}")
        print("Continuing without BERT features (using zeros)...")
        # Return df with dummy BERT columns filled with zeros to maintain shape
        bert_dim = 768  # Standard BERT dimension
        for i in range(bert_dim):
            df[f"bert_{i}"] = 0.0
        return df

def add_user_book_embeddings(df, interactions_df, user_col="user_id", book_col="book_id", rating_col="rating", cfg=None):
    embedder = UserBookEmbedder(no_components=cfg.get("lfm_dim", 32))
    embedder.fit(interactions_df, user_col=user_col, book_col=book_col, rating_col=rating_col)
    user_emb, book_emb = embedder.get_embedding_dfs()
    # Convert merge keys to same type for reliable merging
    # LightFM uses string IDs, but df might have int/float IDs - convert both to string for consistency
    if user_col in df.columns and user_col in user_emb.columns:
        df[user_col] = df[user_col].astype(str)
        user_emb[user_col] = user_emb[user_col].astype(str)
    if book_col in df.columns and book_col in book_emb.columns:
        df[book_col] = df[book_col].astype(str)
        book_emb[book_col] = book_emb[book_col].astype(str)
    df = df.merge(user_emb, on=user_col, how="left")
    df = df.merge(book_emb, on=book_col, how="left")
    return df
# ---


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    """Fills missing values using a defined strategy.

    Fills missing values for age, aggregated features, and categorical features
    to prepare the DataFrame for model training. Uses metrics from the training
    set (e.g., global mean) to fill NaNs.

    Args:
        df (pd.DataFrame): The DataFrame with missing values.
        train_df (pd.DataFrame): The training data, used for calculating fill metrics.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    print("Handling missing values...")

    # Calculate global mean from training data for filling
    global_mean = train_df[config.TARGET].mean()

    # Fill age with the median
    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

    # Fill aggregate features for "cold start" users/items (only if they exist)
    if constants.F_USER_MEAN_RATING in df.columns:
        df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    if constants.F_BOOK_MEAN_RATING in df.columns:
        df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)

    if constants.F_USER_RATINGS_COUNT in df.columns:
        df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    if constants.F_BOOK_RATINGS_COUNT in df.columns:
        df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)

    # Fill missing avg_rating from book_data with global mean
    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    # Fill genre counts with 0
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    # Fill TF-IDF features with 0 (for books without descriptions)
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    # Fill BERT features with 0 (for books without descriptions)
    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)

    # Fill remaining categorical features with a special value
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    return df


def create_features(
    df: pd.DataFrame, book_genres_df: pd.DataFrame, descriptions_df: pd.DataFrame, include_aggregates: bool = False, interactions_df=None, cfg=None
) -> pd.DataFrame:
    print("Starting feature engineering pipeline [ENHANCED]...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    
    try:
        if include_aggregates:
            print("  [Step 1/6] Adding aggregate features...")
            df = add_aggregate_features(df, train_df)
        
        print("  [Step 2/6] Adding genre features...")
        df = add_genre_features(df, book_genres_df)
        
        print("  [Step 3/6] Adding TF-IDF features...")
        df = add_text_features_new(df, desc_col=constants.COL_DESCRIPTION, fit_on=train_df, cfg=cfg)
        
        print("  [Step 4/6] Adding BERT features...")
        df = add_bert_features_new(df, descriptions_df, desc_col=constants.COL_DESCRIPTION, id_col=constants.COL_BOOK_ID, cfg=cfg)
        
        print("  [Step 5/6] Adding user/book embeddings...")
        if interactions_df is not None and not (cfg and cfg.get("skip_lfm", False)):
            df = add_user_book_embeddings(df, interactions_df, user_col=constants.COL_USER_ID, book_col=constants.COL_BOOK_ID, rating_col=constants.COL_TARGET, cfg=cfg)
        elif cfg and cfg.get("skip_lfm", False):
            print("    Skipping LightFM (skip_lfm=True)")
            lfm_dim = cfg.get("lfm_dim", 32)
            for i in range(lfm_dim):
                df[f"lfm_u_{i}"] = 0.0
                df[f"lfm_b_{i}"] = 0.0
        
        print("  [Step 6/6] Handling missing values...")
        df = handle_missing_values(df, train_df)
        
        for col in config.CAT_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype("category")
        print("Feature engineering complete [ENHANCED].")
        return df
    except Exception as e:
        import traceback
        print(f"\nERROR in create_features at step shown above:")
        traceback.print_exc()
        raise
