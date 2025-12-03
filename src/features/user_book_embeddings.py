"""
User-Book Embedding Generation via LightFM Collaborative Filtering

- Matrix factorization for user and item embeddings
- Exports .npy/CSV or DataFrame for use in LightGBM/NN ensembles
- Flexible for explicit or implicit feedback settings

Example usage:
    from features.user_book_embeddings import UserBookEmbedder
    embedder = UserBookEmbedder(no_components=64)
    embedder.fit(interaction_df)  # has columns ['user_id','book_id','rating']
    user_emb, book_emb = embedder.get_embeddings()
"""

from pathlib import Path
import numpy as np
import pandas as pd
try:
    from lightfm import LightFM
    from lightfm.data import Dataset
except ImportError:
    raise ImportError("LightFM required! Install via pip install lightfm")
import joblib

class UserBookEmbedder:
    def __init__(self, no_components=32, loss="warp", random_state=42):
        self.no_components = no_components
        self.loss = loss
        self.random_state = random_state
        self.model = None
        self.dataset = None
        self.user_ids = None
        self.book_ids = None
    def fit(self, df: pd.DataFrame, user_col: str = "user_id", book_col: str = "book_id", rating_col: str = "rating"):
        # Convert to string consistently - handle float IDs by converting to int first if possible
        def to_str_consistent(val):
            # Try to convert float to int first to avoid "281.0" vs "281" mismatch
            if pd.isna(val):
                return str(val)
            try:
                if isinstance(val, float) and val.is_integer():
                    return str(int(val))
            except (AttributeError, ValueError):
                pass
            return str(val)
        
        # Extract unique IDs with consistent string conversion
        uids = [to_str_consistent(uid) for uid in df[user_col].dropna().unique()]
        bids = [to_str_consistent(bid) for bid in df[book_col].dropna().unique()]
        
        self.dataset = Dataset()
        self.dataset.fit(users=uids, items=bids)
        
        # Build interactions list with the same consistent string conversion
        interactions_list = []
        for _, row in df[[user_col, book_col, rating_col]].iterrows():
            uid = to_str_consistent(row[user_col])
            bid = to_str_consistent(row[book_col])
            rating = float(row[rating_col]) if pd.notna(row[rating_col]) else 0.0
            interactions_list.append((uid, bid, rating))
        
        (interactions, weights) = self.dataset.build_interactions(interactions_list)
        self.user_ids = uids
        self.book_ids = bids
        self.model = LightFM(no_components=self.no_components, loss=self.loss, random_state=self.random_state)
        self.model.fit(interactions, sample_weight=weights, epochs=10, num_threads=4)
        return self
    def get_embeddings(self):
        if not self.model:
            raise ValueError("Model not fit yet!")
        user_emb = self.model.get_user_representations()[1]  # returns (ids, array)
        book_emb = self.model.get_item_representations()[1]
        return user_emb, book_emb
    def get_embedding_dfs(self):
        user_emb, book_emb = self.get_embeddings()
        user_df = pd.DataFrame(user_emb, index=self.user_ids, columns=[f"lfm_u_{i}" for i in range(user_emb.shape[1])]).reset_index().rename(columns={"index": "user_id"})
        book_df = pd.DataFrame(book_emb, index=self.book_ids, columns=[f"lfm_b_{i}" for i in range(book_emb.shape[1])]).reset_index().rename(columns={"index": "book_id"})
        return user_df, book_df
    def save(self, filepath: str | Path):
        joblib.dump((self.model, self.dataset, self.user_ids, self.book_ids, self.no_components, self.loss), filepath)
    @classmethod
    def load(cls, filepath: str | Path):
        model, dataset, user_ids, book_ids, no_components, loss = joblib.load(filepath)
        obj = cls(no_components=no_components, loss=loss)
        obj.model = model
        obj.dataset = dataset
        obj.user_ids = user_ids
        obj.book_ids = book_ids
        return obj
