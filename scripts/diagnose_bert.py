import numpy as np
import pandas as pd
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from config import DATA_DIR, MODELS_DIR, TEST_SIZE, RANDOM_STATE, MBTI_TYPES
from sklearn.model_selection import train_test_split
from src.features import build_all_features
import joblib

# Load and split
df = pd.read_parquet(Path(DATA_DIR) / "processed_data.parquet")
y = df['label_idx'].values
df_train, df_test, y_train, y_test = train_test_split(
    df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Build features
X_bert_train, X_tfidf_train, X_bert_test, X_tfidf_test, vec, svd = build_all_features(df_train, df_test)

print(f'X_bert_train stats: min={X_bert_train.min():.4f}, max={X_bert_train.max():.4f}, mean={X_bert_train.mean():.4f}, std={X_bert_train.std():.4f}')
print(f'X_bert_test stats: min={X_bert_test.min():.4f}, max={X_bert_test.max():.4f}, mean={X_bert_test.mean():.4f}, std={X_bert_test.std():.4f}')

# Load MLP model
mlp = joblib.load(Path(MODELS_DIR) / 'mlp_bert.pkl')
scaler = joblib.load(Path(MODELS_DIR) / 'bert_scaler.pkl')

# Test on training data (should be high if no overfitting)
X_train_scaled = scaler.transform(X_bert_train)
train_acc = mlp.score(X_train_scaled, y_train)
print(f'MLP train accuracy: {train_acc:.4f}')

# Test on test data
X_test_scaled = scaler.transform(X_bert_test)
test_acc = mlp.score(X_test_scaled, y_test)
print(f'MLP test accuracy: {test_acc:.4f}')

# Per-class accuracy
pred_test = mlp.predict(X_test_scaled)
for i, mbti in enumerate(MBTI_TYPES):
    mask = y_test == i
    if mask.sum() > 0:
        acc = (pred_test[mask] == y_test[mask]).mean()
        print(f'{mbti}: {acc:.3f} ({mask.sum()} samples)')

print(f'\nOverfitting gap: train={train_acc:.4f} - test={test_acc:.4f} = {train_acc - test_acc:.4f}')
