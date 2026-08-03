from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from config import DATA_DIR, TEST_SIZE, RANDOM_STATE, MODELS_DIR
from src.features import build_all_features
from sklearn.model_selection import train_test_split

p_data = Path(DATA_DIR) / "processed_data.parquet"
print('processed data:', p_data, flush=True)

df = pd.read_parquet(p_data)
y = df['label_idx'].values
print('rows total', len(df), flush=True)

# split
from sklearn.model_selection import train_test_split

df_train, df_test, _, y_test = train_test_split(
    df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print('train/test sizes', len(df_train), len(df_test), flush=True)

# build features with error handling
try:
    print('building features...', flush=True)
    X_bert_train, X_tfidf_train, X_bert_test, X_tfidf_test, vec, svd = build_all_features(df_train, df_test)
    print('X_bert_train shape', X_bert_train.shape, flush=True)
    print('X_bert_test shape', X_bert_test.shape, flush=True)
except Exception as e:
    print('feature build failed:', repr(e))
    raise

# load scaler
scaler_path = Path(MODELS_DIR) / 'bert_scaler.pkl'
print('scaler path', scaler_path.exists(), scaler_path, flush=True)
scaler = joblib.load(scaler_path)

print('scaler attrs: n_samples_seen_', getattr(scaler, 'n_samples_seen_', None), flush=True)
print('scaler mean_ len', None if getattr(scaler, 'mean_', None) is None else len(scaler.mean_), flush=True)

# print a few means before scaling
print('train mean (first 5)', np.mean(X_bert_train, axis=0)[:5], flush=True)
print('test mean (first 5)', np.mean(X_bert_test, axis=0)[:5], flush=True)

# after scaling
X_train_scaled = scaler.transform(X_bert_train)
X_test_scaled = scaler.transform(X_bert_test)
print('train scaled mean (first 5)', np.mean(X_train_scaled, axis=0)[:5], flush=True)
print('test scaled mean (first 5)', np.mean(X_test_scaled, axis=0)[:5], flush=True)
print('train scaled std (first 5)', np.std(X_train_scaled, axis=0)[:5], flush=True)
print('test scaled std (first 5)', np.std(X_test_scaled, axis=0)[:5], flush=True)

# inspect mlp model
mlp_path = Path(MODELS_DIR) / 'mlp_bert.pkl'
print('mlp exists', mlp_path.exists(), flush=True)
if mlp_path.exists():
    mlp = joblib.load(mlp_path)
    try:
        print('mlp classes len', len(mlp.classes_), flush=True)
    except Exception as e:
        print('could not read mlp.classes_', e)

print('done', flush=True)
