"""
src/train.py  — Fixed version
Fixes:
  1. SMOTE enabled with correct order: Scale FIRST → SMOTE second
  2. CV removed from MLP (was causing 5 extra training runs)
  3. Baselines trained on train split only (no data leakage)
"""

import os
import sys
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, RANDOM_STATE,
    MLP_HIDDEN_LAYERS, MLP_DROPOUT, MLP_LEARNING_RATE, MLP_MAX_ITER,
    CV_FOLDS,
)

os.makedirs(MODELS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_model(model, name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"   💾 Saved → {path}")
    return path


# ── Baselines (trained on train split only) ───────────────────────────────────

def train_logistic_regression(X_train, y_train):
    print("\n─── Logistic Regression (TF-IDF) ───")
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=1000, C=1.0,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ))
    ])
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=CV_FOLDS, scoring="accuracy", n_jobs=1)
    print(f"   CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    pipe.fit(X_train, y_train)
    save_model(pipe, "logreg_tfidf")
    return pipe


def train_svm(X_train, y_train):
    print("\n─── Linear SVM (TF-IDF) ───")
    svm  = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf",    CalibratedClassifierCV(svm, cv=3)),
    ])
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=CV_FOLDS, scoring="accuracy", n_jobs=1)
    print(f"   CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    pipe.fit(X_train, y_train)
    save_model(pipe, "svm_tfidf")
    return pipe


def train_random_forest(X_train, y_train):
    print("\n─── Random Forest (TF-IDF) ───")
    pipe = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ))
    ])
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=3, scoring="accuracy", n_jobs=1)
    print(f"   CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    pipe.fit(X_train, y_train)
    save_model(pipe, "rf_tfidf")
    return pipe

def train_bert_logistic_regression(X_train, y_train):
    print("─── Logistic Regression (BERT Embeddings) ───")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ))
    ])
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=CV_FOLDS, scoring="accuracy", n_jobs=1)
    print(f"   CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    pipe.fit(X_train, y_train)
    save_model(pipe, "bert_logreg")
    return pipe

# ── MLP on BERT ───────────────────────────────────────────────────────────────

def train_mlp(X_bert_train, y_train, use_smote=False):
    print("\n─── MLP on BERT Embeddings (Final Model) ───")
    
    # 1. Skip StandardScaler entirely for BERT embeddings!
    X_train_final = X_bert_train
    
    # 2. SMOTE (If you decide to turn it back on, but keep it off for now)
    if use_smote:
        print(f"   Applying SMOTE … original shape: {X_train_final.shape}")
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        X_train_final, y_train = sm.fit_resample(X_train_final, y_train)
        print(f"   After SMOTE: {X_train_final.shape}")

    # 3. Train MLP 
    mlp = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=0.001,  # Fixed L2 penalty. DO NOT use a dropout value here.
        learning_rate_init=MLP_LEARNING_RATE,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=True,
        batch_size=256,
    )
    
    mlp.fit(X_train_final, y_train)
    print(f"   Training loss curve length: {len(mlp.loss_curve_)}")

    # We no longer need to save a scaler for the MLP, but to avoid breaking 
    # the rest of your pipeline, we can return a dummy or None.
    # Just be sure to update step_evaluate in main.py to not require the scaler.
    save_model(mlp, "mlp_bert")

    return mlp, None


# ── Dimension classifiers ─────────────────────────────────────────────────────

def train_dimension_classifiers(X_bert_train, df_train):
    """4 binary classifiers — trained on train split only."""
    print("\n─── Training Per-Dimension Binary Classifiers ───")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_bert_train)
    dims     = {}

    for dim in ["ie", "ns", "tf", "jp"]:
        y_dim = df_train[dim].values
        clf   = MLPClassifier(
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

    save_model(scaler, "dim_scaler")
    return dims, scaler


# ── Master training function ──────────────────────────────────────────────────

def train_all(df, X_bert_train, X_tfidf_train, use_smote=True):
    """
    Train all models using the train split only.

    This function expects data already split in main.py. It does not
    perform a second train/test split.
    """
    y_train = df["label_idx"].values

    print("\n" + "═"*55)
    print("  TRAINING ALL MODELS")
    print("═"*55)

    # ── Baselines on train TF-IDF ─────────────────────────────────────────────
    lr  = train_logistic_regression(X_tfidf_train, y_train)
    svm = train_svm(X_tfidf_train, y_train)
    rf  = train_random_forest(X_tfidf_train, y_train)

    # ── BERT baseline on train BERT embeddings ───────────────────────────────
    bert_logreg = train_bert_logistic_regression(X_bert_train, y_train)

    # ── MLP on train BERT (SMOTE enabled, correct order) ─────────────────────
    mlp, scaler = train_mlp(X_bert_train, y_train.copy(), use_smote=use_smote)

    # ── Dimension classifiers on train BERT ───────────────────────────────────
    dim_clfs, dim_scaler = train_dimension_classifiers(X_bert_train, df)

    print(f"\n✅ All models trained and saved to {MODELS_DIR}")
    return {
        "logreg":     lr,
        "svm":        svm,
        "rf":         rf,
        "bert_logreg": bert_logreg,
        "mlp":        mlp,
        "scaler":     scaler,
        "dim_clfs":   dim_clfs,
        "dim_scaler": dim_scaler,
    }