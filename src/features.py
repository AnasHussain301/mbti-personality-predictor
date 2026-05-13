"""
Feature extraction helpers.

The original main.py talks about BERT embeddings. To keep this project fully
local and reproducible, we expose a dense embedding matrix made from TF-IDF +
TruncatedSVD in the same X_bert slot.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from config import MODELS_DIR, RANDOM_STATE, SVD_COMPONENTS, TFIDF_MAX_FEATURES


def build_all_features(df: pd.DataFrame):
    """
    Build TF-IDF + SVD dense embeddings
    """

    if "clean_posts" not in df.columns:
        raise ValueError("Expected preprocessed dataframe with a 'clean_posts' column")

    texts = df["clean_posts"].fillna("")

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    # Sparse TF-IDF matrix
    X_tfidf = vectorizer.fit_transform(texts)

    # Dense embeddings using SVD
    n_components = min(SVD_COMPONENTS, max(2, min(X_tfidf.shape) - 1))

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=RANDOM_STATE
    )

    X_dense = svd.fit_transform(X_tfidf).astype(np.float32)

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(
        vectorizer,
        os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    )

    joblib.dump(
        svd,
        os.path.join(MODELS_DIR, "svd_embedder.pkl")
    )

    print(f"   TF-IDF shape: {X_tfidf.shape}")
    print(f"   Dense embedding shape: {X_dense.shape}")

    return X_dense, X_tfidf, vectorizer


def transform_texts(texts, vectorizer=None, svd=None):
    """
    Transform new texts using saved TF-IDF + SVD pipeline
    """

    if vectorizer is None:
        vectorizer = joblib.load(
            os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        )

    if svd is None:
        svd = joblib.load(
            os.path.join(MODELS_DIR, "svd_embedder.pkl")
        )

    # TF-IDF transform
    X_tfidf = vectorizer.transform(texts)

    # SVD dense transform
    X_dense = svd.transform(X_tfidf).astype(np.float32)

    return X_dense, X_tfidf


def embed_single(text):
    """
    Create embedding for one text sample
    Compatible with old predict.py logic
    """

    X_dense, _ = transform_texts([text])

    return X_dense


def extract_vader_single(text):
    """
    Placeholder sentiment features.
    Keeps compatibility with old prediction pipeline.
    """

    return np.zeros((1, 4), dtype=np.float32)