"""
src/predict.py
──────────────
Inference module — loads all saved models and runs prediction
on new user text in under 3 seconds.

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
    Full inference pipeline for a single user text input.
    """

    t0 = time.time()

    # 1. Preprocess
    from src.preprocess import preprocess_single
    clean = preprocess_single(text)

    # 2. Embedding (must be 256-dim)
    from src.features import embed_single
    X = embed_single(clean)   # Expected shape: (1, 256)

    # Safety check (prevents your previous error)
    if X.shape[1] != 256:
        raise ValueError(f"Embedding size mismatch: got {X.shape[1]}, expected 256")

    # 3. Scale
    scaler = _load("bert_scaler")
    X_scaled = scaler.transform(X)

    # 4. Predict MBTI type
    mlp = _load("mlp_bert")

    label_idx = int(mlp.predict(X_scaled)[0])
    mbti_type = MBTI_TYPES[label_idx]

    # Probabilities
    proba = mlp.predict_proba(X_scaled)[0]

    # ✅ Confidence (FIX)
    confidence = float(proba[label_idx])

    # Top-3 predictions
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [(MBTI_TYPES[i], float(proba[i])) for i in top3_idx]

    # 5. Dimension classifiers
    dim_scaler = _load("dim_scaler")
    X_dim = dim_scaler.transform(X)

    dim_scores = {}

    dim_labels = {
        "ie": ("I", "E"),
        "ns": ("N", "S"),
        "tf": ("T", "F"),
        "jp": ("J", "P"),
    }

    for idx, (dim, (neg_label, pos_label)) in enumerate(dim_labels.items()):

        clf = _load(f"mlp_{dim}")
        prob = clf.predict_proba(X_dim)[0]

        pred_letter = mbti_type[idx]

        # Select correct probability
        if pred_letter == neg_label:
            score = float(prob[0])
        else:
            score = float(prob[1])

        key = f"{neg_label}/{pos_label}"
        dim_scores[key] = round(score, 3)

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
        "confidence": confidence,   # ✅ FIX ADDED
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

    print(f"\nConfidence: {result['confidence']:.2%}")

    print(f"\nDimension Confidence:")
    for k, v in result['dimension_scores'].items():
        print(f"  {k:5s}: {v:.2%}")

    print(f"\nTop 3 Predictions:")
    for t, p in result['top3']:
        print(f"  {t}: {p:.2%}")

    print(f"\nInference time: {result['inference_ms']} ms")