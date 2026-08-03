"""
src/predict.py
──────────────
Inference module — loads the 4 binary dimension models and runs an ensemble 
prediction on new user text in under 3 seconds.

Used by the Streamlit app.
"""

import os
import sys
import time
import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, MBTI_TYPES, MBTI_DESCRIPTIONS

# ── Lazy model cache (loaded once on first call) ─────────────────────────────

_models = {}

def _load(name: str):
    if name not in _models:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model '{name}.pkl' not found in {MODELS_DIR}. "
                f"Run main.py to train first."
            )
        _models[name] = joblib.load(path)
    return _models[name]


# ── Single prediction ─────────────────────────────────────────────────────────

def predict_mbti(text: str, return_proba: bool = True) -> dict:
    """
    Full inference pipeline for a single user text input using the 4-model ensemble.
    """
    t0 = time.time()

    # 1. Preprocess
    from src.preprocess import preprocess_single
    clean = preprocess_single(text)

    # 2. Embedding (Sentence Transformers all-MiniLM-L6-v2 outputs 384 dimensions)
    from src.features import embed_single
    X = embed_single(clean)   # Expected shape: (1, 384)

    # Safety check
    if X.shape[1] != 384:
        raise ValueError(f"Embedding size mismatch: got {X.shape[1]}, expected 384")

    # Note: We NO LONGER apply StandardScaler to BERT embeddings.
    
    # 3. Predict Dimensions independently
    dim_labels = {
        "ie": ("I", "E"),
        "ns": ("N", "S"),
        "tf": ("T", "F"),
        "jp": ("J", "P"),
    }

    dim_probs = {}
    mbti_type = ""
    dim_scores = {}

    for dim, (neg_label, pos_label) in dim_labels.items():
        clf = _load(f"mlp_{dim}")
        
        # Predict probabilities directly on the raw BERT embedding
        prob = clf.predict_proba(X)[0] 
        dim_probs[dim] = prob
        
        # prob[0] corresponds to index 0 (I, N, T, J)
        # prob[1] corresponds to index 1 (E, S, F, P)
        if prob[0] > prob[1]:
            mbti_type += neg_label
            dim_scores[f"{neg_label}/{pos_label}"] = float(prob[0])
        else:
            mbti_type += pos_label
            dim_scores[f"{neg_label}/{pos_label}"] = float(prob[1])

    # 4. Calculate Joint Probabilities for all 16 types to find Top 3
    all_types_probs = []
    for t in MBTI_TYPES:
        p_ie = dim_probs["ie"][0] if t[0] == "I" else dim_probs["ie"][1]
        p_ns = dim_probs["ns"][0] if t[1] == "N" else dim_probs["ns"][1]
        p_tf = dim_probs["tf"][0] if t[2] == "T" else dim_probs["tf"][1]
        p_jp = dim_probs["jp"][0] if t[3] == "J" else dim_probs["jp"][1]
        
        # The overall probability of an MBTI type is the product of its 4 independent dimensions
        overall_prob = float(p_ie * p_ns * p_tf * p_jp)
        all_types_probs.append((t, overall_prob))

    # Sort descending by probability
    all_types_probs.sort(key=lambda x: x[1], reverse=True)
    
    top3 = all_types_probs[:3]
    confidence = top3[0][1] # The probability of the #1 predicted type

    t1 = time.time()

    title, description = MBTI_DESCRIPTIONS.get(
        mbti_type,
        ("Unknown Type", "No description available.")
    )

    return {
        "mbti_type": mbti_type,
        "title": title,
        "description": description,
        "dimension_scores": dim_scores,
        "top3": top3,
        "confidence": confidence,
        "inference_ms": int((t1 - t0) * 1000),
        "clean_text": clean,
    }


# ── Batch prediction ─────────────────────────────────────────────────────────

def batch_predict(texts: list) -> list:
    return [predict_mbti(t) for t in texts]


# ── Test run ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = """
    I spent the whole evening reading philosophy and writing in my journal.
    I prefer deep one-on-one conversations over big social gatherings.
    Sometimes I feel like nobody truly understands how I see the world.
    I find beauty in abstract ideas and love exploring the 'what-ifs' of life.
    """

    result = predict_mbti(sample)

    print(f"\nPredicted Type : {result['mbti_type']} — {result['title']}")
    print(f"Description    : {result['description']}")

    print(f"\nOverall Joint Confidence: {result['confidence']:.2%}")

    print(f"\nDimension Confidence:")
    for k, v in result['dimension_scores'].items():
        print(f"  {k:5s}: {v:.2%}")

    print(f"\nTop 3 Predictions:")
    for t, p in result['top3']:
        print(f"  {t}: {p:.2%}")

    print(f"\nInference time: {result['inference_ms']} ms")