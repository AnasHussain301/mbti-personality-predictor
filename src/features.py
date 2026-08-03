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

# Optional: use sentence-transformers for real semantic embeddings (BERT-like)
try:
    from sentence_transformers import SentenceTransformer
    import torch
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False


def build_all_features(df_train: pd.DataFrame, df_test: pd.DataFrame = None):
    """
    Build TF-IDF + SVD dense embeddings from training text, then transform test data.
    """

    text_col = "clean_text" if "clean_text" in df_train.columns else "clean_posts"
    texts_train = df_train[text_col].fillna("")

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    # Sparse TF-IDF matrix
    X_tfidf_train = vectorizer.fit_transform(texts_train)

    # Dense embeddings using either sentence-transformers (recommended)
    # or TruncatedSVD fallback for TF-IDF if sentence-transformers isn't available.
    if _HAS_SENTENCE_TRANSFORMERS:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = "all-MiniLM-L6-v2"
        print(f"   Using Sentence-Transformers model '{model_name}' on device={device}")
        bert = SentenceTransformer(model_name, device=device)
        print(f"   Encoding {len(texts_train)} training texts...")
        # encode returns numpy arrays
        X_dense_train = bert.encode(
            texts_train.tolist(), batch_size=32, show_progress_bar=True, convert_to_numpy=True
        ).astype(np.float32)
        print(f"   ✓ Training texts encoded")
    else:
        n_components = min(SVD_COMPONENTS, max(2, min(X_tfidf_train.shape) - 1))
        svd = TruncatedSVD(
            n_components=n_components,
            random_state=RANDOM_STATE
        )
        X_dense_train = svd.fit_transform(X_tfidf_train).astype(np.float32)

    X_tfidf_test = None
    X_dense_test = None
    if df_test is not None:
        text_col_test = "clean_text" if "clean_text" in df_test.columns else "clean_posts"
        texts_test = df_test[text_col_test].fillna("")
        X_tfidf_test = vectorizer.transform(texts_test)
        if _HAS_SENTENCE_TRANSFORMERS:
            # If we used sentence-transformers, encode test texts the same way
            print(f"   Encoding {len(texts_test)} test texts...")
            X_dense_test = bert.encode(
                texts_test.tolist(), batch_size=32, show_progress_bar=True, convert_to_numpy=True
            ).astype(np.float32)
            print(f"   ✓ Test texts encoded")
        else:
            X_dense_test = svd.transform(X_tfidf_test).astype(np.float32)

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(
        vectorizer,
        os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    )

    # Save SVD only if it was created (fallback case)
    if not _HAS_SENTENCE_TRANSFORMERS:
        joblib.dump(
            svd,
            os.path.join(MODELS_DIR, "svd_embedder.pkl")
        )

    print(f"   TF-IDF train shape: {X_tfidf_train.shape}")
    print(f"   Dense train shape: {X_dense_train.shape}")
    if X_tfidf_test is not None:
        print(f"   TF-IDF test shape: {X_tfidf_test.shape}")
        print(f"   Dense test shape: {X_dense_test.shape}")

    # Return the dense embeddings (X_bert_*), TF-IDF matrices, and the vectorizer/svd
    return X_dense_train, X_tfidf_train, X_dense_test, X_tfidf_test, vectorizer, (svd if not _HAS_SENTENCE_TRANSFORMERS else None)


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


# Add this global variable right above embed_single
_bert_model = None

def embed_single(text):
    """
    Create embedding for one text sample.
    Uses SentenceTransformers (BERT) to match the 384-dimension training setup.
    """
    global _bert_model
    
    if _HAS_SENTENCE_TRANSFORMERS:
        if _bert_model is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _bert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            
        # Generate the 384-dimensional BERT embedding
        X_dense = _bert_model.encode([text], convert_to_numpy=True).astype(np.float32)
        return X_dense
    else:
        # Fallback ONLY if sentence-transformers isn't installed
        X_dense, _ = transform_texts([text])
        return X_dense


def extract_vader_single(text):
    """
    Placeholder sentiment features.
    Keeps compatibility with old prediction pipeline.
    """

    return np.zeros((1, 4), dtype=np.float32)