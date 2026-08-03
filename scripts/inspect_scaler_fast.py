from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from config import DATA_DIR, TEST_SIZE, RANDOM_STATE, MODELS_DIR
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

# load vectorizer and svd
vec = joblib.load(Path(MODELS_DIR)/'tfidf_vectorizer.pkl')
svd = joblib.load(Path(MODELS_DIR)/'svd_embedder.pkl')
print('loaded vectorizer and svd', flush=True)

text_col = 'clean_text' if 'clean_text' in df_train.columns else 'clean_posts'
texts_train = df_train[text_col].fillna("")
texts_test = df_test[text_col].fillna("")
X_tfidf_train = vec.transform(texts_train)
X_tfidf_test = vec.transform(texts_test)
X_bert_train = svd.transform(X_tfidf_train).astype(np.float32)
X_bert_test = svd.transform(X_tfidf_test).astype(np.float32)
print('X_bert_train shape', X_bert_train.shape, flush=True)
print('X_bert_test shape', X_bert_test.shape, flush=True)

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
        print('could not read mlp.classes_', e, flush=True)

print('done', flush=True)
