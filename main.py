"""
main.py
───────
Master training pipeline for the MBTI Personality Predictor.

Usage:
    python main.py                   # Full pipeline (EDA + Train + Eval)
    python main.py --skip-eda        # Skip EDA, go straight to training
    python main.py --eval-only       # Load saved models and evaluate only
    python main.py --predict         # Quick test prediction after training

The trained models are saved to ./models/ and the Streamlit app
can then be launched with:
    streamlit run app/app.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_PATH, MODELS_DIR, DATA_DIR, OUTPUTS_DIR,
    MBTI_TYPES, TEST_SIZE, RANDOM_STATE,
)

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)

# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MBTI Personality Predictor — Training Pipeline")
    parser.add_argument("--skip-eda",    action="store_true", help="Skip EDA step")
    parser.add_argument("--eval-only",   action="store_true", help="Only evaluate saved models")
    parser.add_argument("--predict",     action="store_true", help="Run a test prediction after training")
    parser.add_argument("--no-smote",    action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--data",        type=str, default=RAW_DATA_PATH, help="Path to mbti_1.csv")
    return parser.parse_args()


# ── Pipeline Steps ────────────────────────────────────────────────────────────

def step_load_data(data_path: str) -> pd.DataFrame:
    print("\n" + "═"*55)
    print("  STEP 1: Load Data")
    print("═"*55)

    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at: {data_path}")
        print("\n👉 Download it from: https://www.kaggle.com/datasets/datasnaek/mbti-type")
        print(f"   Place the CSV at: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} rows, columns: {list(df.columns)}")
    print(f"   Type distribution:\n{df['type'].value_counts().to_string()}")
    return df


def step_eda(df: pd.DataFrame):
    print("\n" + "═"*55)
    print("  STEP 2: Exploratory Data Analysis")
    print("═"*55)
    from src.eda import run_full_eda
    run_full_eda(df)


def step_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═"*55)
    print("  STEP 3: Preprocessing")
    print("═"*55)
    from src.preprocess import preprocess_dataframe
    df_clean = preprocess_dataframe(df, verbose=True)

    # Cache to disk
    cache_path = os.path.join(DATA_DIR, "processed_data.parquet")
    df_clean.to_parquet(cache_path, index=False)
    print(f"   Cached → {cache_path}")
    return df_clean


def step_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    print("\n" + "═"*55)
    print("  STEP 4: Feature Extraction")
    print("═"*55)
    from src.features import build_all_features
    X_bert_train, X_tfidf_train, X_bert_test, X_tfidf_test, tfidf_vec, svd_model = build_all_features(
        df_train,
        df_test,
    )
    return X_bert_train, X_tfidf_train, X_bert_test, X_tfidf_test, tfidf_vec, svd_model


def step_train(df, X_bert, X_tfidf, use_smote: bool = True):
    print("\n" + "═"*55)
    print("  STEP 5: Model Training")
    print("═"*55)
    from src.train import train_all
    models = train_all(df, X_bert, X_tfidf, use_smote=use_smote)
    return models


def step_evaluate(y_test, X_bert_test, X_tfidf_test, models: dict):
    print("\n" + "═"*55)
    print("  STEP 6: Evaluation")
    print("═"*55)
    from src.evaluate import (
        evaluate_model, evaluate_dimensions,
        build_comparison_table, plot_confusion_matrix,
        plot_f1_per_class, plot_mlp_loss_curve,
        plot_model_comparison,
    )

    all_results = []

    # Baselines (TF-IDF)
    for name, key in [("Logistic Regression", "logreg"), ("SVM", "svm"), ("Random Forest", "rf")]:
        clf = models[key]
        pred = clf.predict(X_tfidf_test)
        res = evaluate_model(y_test, pred, model_name=name)
        all_results.append(res)

    # BERT baseline (logistic regression on embeddings)
    bert_logreg = models.get("bert_logreg")
    if bert_logreg is not None:
        X_bert_scaled = StandardScaler().fit(models["scaler"].transform(X_bert_test)) if False else None
        pred_bert_logreg = bert_logreg.predict(X_bert_test)
        res_bert_logreg = evaluate_model(y_test, pred_bert_logreg, model_name="BERT LogReg")
        all_results.append(res_bert_logreg)

    # MLP (BERT)
   # MLP (BERT)
    mlp      = models["mlp"]
    # No scaler needed for raw BERT embeddings
    pred_mlp = mlp.predict(X_bert_test)
    res_mlp  = evaluate_model(y_test, pred_mlp, model_name="MLP (BERT)")
    all_results.append(res_mlp)

    # Per-dimension accuracy for MLP
    print("\n─── Per-Dimension Accuracy (MLP) ───")
    true_types = [MBTI_TYPES[i] for i in y_test]
    pred_types = [MBTI_TYPES[i] for i in pred_mlp]
    evaluate_dimensions(true_types, pred_types)

    # Comparison table
    cmp_df = build_comparison_table(all_results)
    cmp_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"), index=False)

    # Plots
    plot_confusion_matrix(y_test, pred_mlp, model_name="MLP BERT")
    plot_f1_per_class(res_mlp["report"],    model_name="MLP BERT")
    plot_mlp_loss_curve(mlp)
    plot_model_comparison(cmp_df)

    return all_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "="*55)
    print("  MBTI PERSONALITY PREDICTOR — Training Pipeline")
    print("  RTX 4060 | GPU Accelerated BERT Embeddings")
    print("="*55)

    # ── Load data ────────────────────────────────────────────────────────────
    df_raw = step_load_data(args.data)

    # ── EDA ─────────────────────────────────────────────────────────────────
    if not args.skip_eda and not args.eval_only:
        step_eda(df_raw)

    # ── Check for cached preprocessed data ──────────────────────────────────
    cache_path = os.path.join(DATA_DIR, "processed_data.parquet")

    if os.path.exists(cache_path):
         print(f"\n📂 Loading cached preprocessed data from {cache_path}")
         df_clean = pd.read_parquet(cache_path)
         print(f"   {len(df_clean)} rows loaded")
         # Add dimension columns if missing
         if "ie" not in df_clean.columns:
            df_clean["ie"] = df_clean["type"].apply(lambda x: 0 if x[0] == "I" else 1)
            df_clean["ns"] = df_clean["type"].apply(lambda x: 0 if x[1] == "N" else 1)
            df_clean["tf"] = df_clean["type"].apply(lambda x: 0 if x[2] == "T" else 1)
            df_clean["jp"] = df_clean["type"].apply(lambda x: 0 if x[3] == "J" else 1)
            print("   Added dimension columns: ie, ns, tf, jp")
    else:
        df_clean = step_preprocess(df_raw)

    # ── Train/test split ─────────────────────────────────────────────────────
    y = df_clean["label_idx"].values
    df_train, df_test, _, y_test = train_test_split(
        df_clean,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ── Features ─────────────────────────────────────────────────────────────
    X_bert_train, X_tfidf_train, X_bert_test, X_tfidf_test, tfidf_vec, svd_model = \
        step_features(df_train, df_test)

    if not args.eval_only:
        # ── Training ─────────────────────────────────────────────────────────
        models = step_train(df_train, X_bert_train, X_tfidf_train, use_smote=not args.no_smote)
    else:
        # ── Load pre-trained models ───────────────────────────────────────────
        import joblib
        print("\n📂 Loading saved models ...")
        models = {
            "logreg": joblib.load(os.path.join(MODELS_DIR, "logreg_tfidf.pkl")),
            "svm":    joblib.load(os.path.join(MODELS_DIR, "svm_tfidf.pkl")),
            "rf":     joblib.load(os.path.join(MODELS_DIR, "rf_tfidf.pkl")),
            "mlp":    joblib.load(os.path.join(MODELS_DIR, "mlp_bert.pkl")),
            "scaler": joblib.load(os.path.join(MODELS_DIR, "bert_scaler.pkl")),
        }

    # ── Evaluation ────────────────────────────────────────────────────────────
    step_evaluate(y_test, X_bert_test, X_tfidf_test, models)

    # ── Quick test prediction ─────────────────────────────────────────────────
    if args.predict:
        print("\n" + "═"*55)
        print("  STEP 7: Test Prediction")
        print("═"*55)
        from src.predict import predict_mbti
        sample_text = """
        I love spending quiet evenings alone, reading philosophy and writing.
        Big social gatherings drain me. I am always thinking about the future
        and what could be rather than what is. Logic guides most of my decisions.
        I prefer having a clear plan rather than leaving things open-ended.
        """
        result = predict_mbti(sample_text)
        print(f"\n  Predicted Type : {result['mbti_type']} — {result['title']}")
        print(f"  Description    : {result['description']}")
        print(f"\n  Dimension Confidence:")
        for k, v in result['dimension_scores'].items():
            bar = "█" * int(v * 20)
            print(f"    {k}: {bar} {v:.1%}")
        print(f"\n  Top 3: {result['top3']}")
        print(f"  Inference: {result['inference_ms']} ms")

    print("\n" + "✅ "*18)
    print("\n  All done! Launch the web app with:")
    print("  👉  streamlit run app/app.py\n")


if __name__ == "__main__":
    main()
