"""
TF-IDF feature generator module.

- Scikit-learn-based for easy fit-transform, tuning, and pickling
- Works on any DataFrame with a text column (book descriptions, etc.)
- Provides save/load interface for efficient reuse in ML pipelines

Example usage:
    from features.tfidf_features import TfidfFeatureGenerator
    tfg = TfidfFeatureGenerator(max_features=500)
    tfg.fit(df, text_col="description")
    arr = tfg.transform(df)
    tfg.save("tfidf_vectorizer.pkl")
    tfg2 = TfidfFeatureGenerator.load("tfidf_vectorizer.pkl")
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

class TfidfFeatureGenerator:
    def __init__(self, max_features=500, min_df=2, max_df=0.95, ngram_range=(1,2), analyzer="word"):
        self.params = dict(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            analyzer=analyzer
        )
        self.vectorizer = TfidfVectorizer(**self.params)
        self.feature_names_ = None
    def fit(self, df: pd.DataFrame, text_col: str) -> None:
        texts = df[text_col].fillna("").astype(str).tolist()
        X = self.vectorizer.fit_transform(texts)
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        return self
    def transform(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        texts = df[text_col].fillna("").astype(str).tolist()
        X = self.vectorizer.transform(texts)
        if self.feature_names_ is None:
            self.feature_names_ = self.vectorizer.get_feature_names_out()
        return pd.DataFrame(X.toarray(), columns=[f"tfidf_{f}" for f in self.feature_names_], index=df.index)
    def fit_transform(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        self.fit(df, text_col)
        return self.transform(df, text_col)
    def save(self, filepath: str | Path):
        joblib.dump((self.vectorizer, self.feature_names_), filepath)
    @classmethod
    def load(cls, filepath: str | Path):
        vec, names = joblib.load(filepath)
        obj = cls()
        obj.vectorizer = vec
        obj.feature_names_ = names
        return obj
