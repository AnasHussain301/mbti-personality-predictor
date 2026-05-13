"""
src/train.py
────────────
Model training module.

Models:
  1. Logistic Regression  (TF-IDF baseline)
  2. SVM Linear           (TF-IDF baseline)
  3. Random Forest        (TF-IDF baseline)
  4. MLP Classifier       (BERT+VADER+Stylo — final model)

Also trains 4 binary classifiers for per-dimension predictions (IE, NS, TF, JP).
"""

import os
import sys
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, TEST_SIZE, RANDOM_STATE,
    MLP_HIDDEN_LAYERS, MLP_DROPOUT, MLP_LEARNING_RATE, MLP_MAX_ITER,
    MBTI_TYPES, CV_FOLDS,
)

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def save_model(model, name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"   💾 Saved → {path}")
    return path


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    return joblib.load(path)


def apply_smote(X, y, random_state=RANDOM_STATE):
    """Apply SMOTE to balance classes. Works on dense arrays."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    print(f"   Applying SMOTE … original shape: {X.shape}")
    sm = SMOTE(random_state=random_state, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"   After SMOTE: {X_res.shape}")
    return X_res, y_res


# ── Baselines ────────────────────────────────────────────────────────────────

def train_logistic_regression(X_tfidf, y, cv: bool = True):
    """Logistic Regression on TF-IDF features."""
    print("\n─── Logistic Regression (TF-IDF) ───")
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])
    if cv:
        scores = cross_val_score(pipe, X_tfidf, y, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
        print(f"   CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    pipe.fit(X_tfidf, y)
    save_model(pipe, "logreg_tfidf")
    return pipe


def train_svm(X_tfidf, y, cv: bool = True):
    """Linear SVM on TF-IDF features."""
    print("\n─── Linear SVM (TF-IDF) ───")
    svm = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    # Wrap with Platt scaling to get probabilities
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", CalibratedClassifierCV(svm, cv=3)),
    ])
    if cv:
        scores = cross_val_score(pipe, X_tfidf, y, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
        print(f"   CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    pipe.fit(X_tfidf, y)
    save_model(pipe, "svm_tfidf")
    return pipe


def train_random_forest(X_tfidf, y, cv: bool = True):
    """Random Forest on TF-IDF features."""
    print("\n─── Random Forest (TF-IDF) ───")
    pipe = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])
    if cv:
        scores = cross_val_score(pipe, X_tfidf, y, cv=3, scoring="accuracy", n_jobs=-1)
        print(f"   CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    pipe.fit(X_tfidf, y)
    save_model(pipe, "rf_tfidf")
    return pipe


# ── MLP on BERT embeddings ────────────────────────────────────────────────────

def train_mlp(X_bert: np.ndarray, y: np.ndarray, use_smote: bool = True, cv: bool = True):
    """
    MLP Classifier trained on fused BERT+VADER+Stylo features.
    This is the primary final model.
    """
    print("\n─── MLP on BERT Embeddings (Final Model) ───")

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_bert)

    if use_smote:
        X_scaled, y = apply_smote(X_scaled, y)

    mlp = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,    # (512, 256)
        activation="relu",
        solver="adam",
        alpha=MLP_DROPOUT,                        # L2 regularization proxy
        learning_rate_init=MLP_LEARNING_RATE,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=True,
        batch_size=256,
    )

    if cv and not use_smote:  # CV before SMOTE to avoid data leakage
        skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(mlp, X_scaled, y, cv=skf, scoring="accuracy", n_jobs=1)
        print(f"   CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    mlp.fit(X_scaled, y)

    # Save both scaler and model
    save_model(scaler, "bert_scaler")
    save_model(mlp,    "mlp_bert")

    print(f"   Training loss curve length: {len(mlp.loss_curve_)}")
    return mlp, scaler


# ── Per-dimension binary classifiers ─────────────────────────────────────────

def train_dimension_classifiers(X_bert: np.ndarray, df_labels) -> dict:
    """
    Train four separate binary classifiers for:
        IE (0=I, 1=E)
        NS (0=N, 1=S)
        TF (0=T, 1=F)
        JP (0=J, 1=P)

    These are used in the Streamlit app to show per-dimension confidence.
    Returns dict: {'ie': clf, 'ns': clf, 'tf': clf, 'jp': clf}
    """
    print("\n─── Training Per-Dimension Binary Classifiers ───")
    dims = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bert)

    for dim in ["ie", "ns", "tf", "jp"]:
        y_dim = df_labels[dim].values
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=200,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            batch_size=256,
        )
        clf.fit(X_scaled, y_dim)
        save_model(clf, f"mlp_{dim}")
        dims[dim] = clf
        print(f"   ✅ {dim.upper()} classifier trained")

    # Scaler shared across dimension classifiers
    save_model(scaler, "dim_scaler")
    return dims, scaler


# ── Master training function ──────────────────────────────────────────────────

def train_all(df, X_bert, X_tfidf, use_smote=True):
    """
    Run full training pipeline for all models.

    Args:
        df:      preprocessed DataFrame (must have label_idx, ie, ns, tf, jp)
        X_bert:  fused BERT feature matrix (N, 388+)
        X_tfidf: TF-IDF sparse matrix (N, 5000)
    """
    y = df["label_idx"].values

    print("\n" + "═"*55)
    print("  TRAINING ALL MODELS")
    print("═"*55)

    # Baselines
    lr  = train_logistic_regression(X_tfidf, y)
    svm = train_svm(X_tfidf, y)
    rf  = train_random_forest(X_tfidf, y)

    # Final model
    mlp, scaler = train_mlp(X_bert, y.copy(), use_smote=use_smote)

    # Dimension classifiers
    dim_clfs, dim_scaler = train_dimension_classifiers(X_bert, df)

    print("\n✅ All models trained and saved to", MODELS_DIR)
    return {
        "logreg": lr,
        "svm":    svm,
        "rf":     rf,
        "mlp":    mlp,
        "scaler": scaler,
        "dim_clfs":   dim_clfs,
        "dim_scaler": dim_scaler,
    }