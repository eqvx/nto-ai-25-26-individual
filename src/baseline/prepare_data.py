"""
Data preparation script that processes raw data and saves it to processed directory.

This script loads raw data, applies filtering (has_read=1), performs feature engineering,
and saves the processed data to data/processed/ for use in training and prediction.
"""

from . import config, constants
from .data_processing import load_and_merge_data
from .features import create_features


def prepare_data() -> None:
    """Processes raw data and saves prepared features to processed directory.

    This function:
    1. Loads raw data from data/raw/
    2. Filters training data (only has_read=1)
    3. Applies feature engineering (genres, TF-IDF, BERT) - NO aggregates to avoid data leakage
    4. Saves processed data to data/processed/processed_features.parquet
    5. Preserves timestamp for temporal splitting

    Note: Aggregate features are computed separately during training to ensure
    temporal correctness (no data leakage from validation set).

    The processed data can then be used by train.py and predict.py without
    re-running the expensive feature engineering steps.
    """
    print("=" * 60)
    print("Data Preparation Pipeline (with advanced features)")
    print("=" * 60)

    # Load and merge raw data
    merged_df, book_genres_df, _, descriptions_df = load_and_merge_data()

    # Merge book descriptions into merged_df to ensure 'description' column exists
    if "description" not in merged_df.columns:
        merged_df = merged_df.merge(descriptions_df[[constants.COL_BOOK_ID, "description"]], on=constants.COL_BOOK_ID, how="left")

    # Prepare interactions DataFrame (for embeddings): use only train, has_read=1, must have user_id/book_id/target
    # We reconstruct this similarly to how train_df is filtered in data_processing
    interactions_df = merged_df[(merged_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN) &
                                (merged_df[constants.COL_TARGET].notna())].copy()
    # Ensure 'description' also present in interactions_df if used
    if "description" not in interactions_df.columns:
        interactions_df = interactions_df.merge(descriptions_df[[constants.COL_BOOK_ID, "description"]], on=constants.COL_BOOK_ID, how="left")
    # Should have columns user_id, book_id, target (rating)

    # Build config for feature generators
    # Reduced batch sizes for stability on Windows/memory-constrained systems
    import os
    is_windows = os.name == 'nt'
    skip_bert_env = os.environ.get("SKIP_BERT", "false").lower() == "true"
    skip_lfm_env = os.environ.get("SKIP_LFM", "false").lower() == "true"

    cfg = {
        "tfidf_max_features": 500,
        "tfidf_min_df": 2,
        "tfidf_max_df": 0.95,
        "tfidf_ngram_range": (1, 2),
        "bert_model": "bert-base-multilingual-cased",
        "bert_batch_size": 2 if is_windows else 4,
        "bert_max_length": 128,
        "lfm_dim": 32,
        "skip_bert": skip_bert_env,
        "skip_lfm": skip_lfm_env,
    }

    print("Configuration:")
    print(f"  - BERT: {'SKIPPED' if cfg['skip_bert'] else 'ENABLED'}")
    print(f"  - LightFM: {'SKIPPED' if cfg['skip_lfm'] else 'ENABLED'}")
    print("  - TF-IDF: ENABLED")

    # Run advanced feature pipeline
    featured_df = create_features(
        merged_df,
        book_genres_df,
        descriptions_df,
        include_aggregates=False,           # Do not add aggregates here, keep leakage-safe!
        interactions_df=None if cfg.get("skip_lfm", False) else interactions_df,
        cfg=cfg
    )

    # Ensure processed directory exists
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    # Save as parquet
    print(f"\nSaving processed data to {processed_path}...")
    featured_df.to_parquet(processed_path, index=False, engine="pyarrow", compression="snappy")
    print("Processed data saved successfully!")

    # Print stats
    train_rows = len(featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN])
    test_rows = len(featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST])
    total_features = len(featured_df.columns)
    print("\nData preparation complete!")
    print(f"  - Train rows: {train_rows:,}")
    print(f"  - Test rows: {test_rows:,}")
    print(f"  - Total features: {total_features}")
    print(f"  - Output file: {processed_path}")


if __name__ == "__main__":
    import sys
    import traceback
    try:
        prepare_data()
    except Exception as e:
        print("\n" + "="*60)
        print("FATAL ERROR - Full Traceback:")
        print("="*60)
        traceback.print_exc()
        print("="*60)
        sys.exit(1)
    except SystemExit:
        raise
    except:
        print("\n" + "="*60)
        print("FATAL ERROR - Unknown exception (possibly C extension crash):")
        print("="*60)
        traceback.print_exc()
        print("="*60)
        sys.exit(1)
