"""
src/preprocess.py
─────────────────
Full text preprocessing pipeline for MBTI social media posts.

Steps:
  1. Remove pipe separators (dataset-specific: posts split by |||)
  2. Remove URLs
  3. Lowercase
  4. Expand MBTI type mentions (remove self-identification bias)
  5. Decode/remove emojis
  6. Remove special characters, numbers, punctuation
  7. Tokenize → Lemmatize → Remove stopwords
  8. Rejoin into clean text
"""

import re
import string
import emoji
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# ── One-time NLTK downloads ──────────────────────────────────────────────────
def download_nltk_resources():
    resources = ["punkt", "wordnet", "stopwords", "omw-1.4", "punkt_tab"]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# ── Constants ────────────────────────────────────────────────────────────────
MBTI_TYPES = [
    "infp", "infj", "intp", "intj", "entp", "enfp", "istp", "isfp",
    "entj", "istj", "enfj", "isfj", "estp", "esfp", "esfj", "estj",
]
MBTI_PATTERN = re.compile(r"\b(" + "|".join(MBTI_TYPES) + r")\b", re.IGNORECASE)

URL_PATTERN  = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)

STOP_WORDS   = set(stopwords.words("english"))
LEMMATIZER   = WordNetLemmatizer()


# ── Individual cleaning functions ────────────────────────────────────────────

def remove_pipe_separators(text: str) -> str:
    """Replace the ||| post separator with space."""
    return text.replace("|||", " ")


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_mbti_mentions(text: str) -> str:
    """
    Remove MBTI type names from posts to avoid the model
    simply learning that users who say 'I am INFP' are INFP.
    """
    return MBTI_PATTERN.sub(" ", text)


def handle_emojis(text: str, mode: str = "remove") -> str:
    """
    mode='remove'  — strip all emojis
    mode='replace' — convert to text description (e.g., 😊 → ':smiling_face:')
    """
    if mode == "replace":
        return emoji.demojize(text, delimiters=(" ", " "))
    return emoji.replace_emoji(text, replace=" ")


def clean_text(text: str) -> str:
    """Lowercase, remove numbers/punctuation/extra spaces."""
    text = text.lower()
    text = re.sub(r"\d+", " ", text)                          # numbers
    text = re.sub(r"[" + re.escape(string.punctuation) + "]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_and_remove_stopwords(text: str) -> str:
    """Tokenize → lemmatize → remove stopwords → rejoin."""
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)


# ── Master pipeline ──────────────────────────────────────────────────────────

def preprocess_single(text: str, emoji_mode: str = "remove") -> str:
    """
    Run the full preprocessing pipeline on a single string.
    Used for real-time inference in the Streamlit app.
    """
    text = remove_pipe_separators(text)
    text = remove_urls(text)
    text = remove_mbti_mentions(text)
    text = handle_emojis(text, mode=emoji_mode)
    text = clean_text(text)
    text = lemmatize_and_remove_stopwords(text)
    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "posts",
    label_col: str = "type",
    emoji_mode: str = "remove",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Preprocess the entire MBTI Kaggle DataFrame.

    Returns a new DataFrame with columns:
        - 'clean_text'    : preprocessed post text
        - 'type'          : original MBTI label (uppercase)
        - 'label_idx'     : integer index (0-15) for the MBTI type
        - 'ie', 'ns', 'tf', 'jp': binary dimension labels (0/1)
    """
    df = df.copy()

    # ── Clean text ───────────────────────────────────────────────────────────
    tqdm.pandas(desc="Preprocessing posts") if verbose else None
    if verbose:
        df["clean_text"] = df[text_col].progress_apply(
            lambda t: preprocess_single(str(t), emoji_mode=emoji_mode)
        )
    else:
        df["clean_text"] = df[text_col].apply(
            lambda t: preprocess_single(str(t), emoji_mode=emoji_mode)
        )

    # ── Labels ───────────────────────────────────────────────────────────────
    df[label_col] = df[label_col].str.upper().str.strip()

    from config import MBTI_TYPES
    label_map = {t: i for i, t in enumerate(MBTI_TYPES)}
    df["label_idx"] = df[label_col].map(label_map)

    # Per-dimension binary labels (useful for dimension-level evaluation)
    df["ie"] = df[label_col].apply(lambda x: 0 if x[0] == "I" else 1)
    df["ns"] = df[label_col].apply(lambda x: 0 if x[1] == "N" else 1)
    df["tf"] = df[label_col].apply(lambda x: 0 if x[2] == "T" else 1)
    df["jp"] = df[label_col].apply(lambda x: 0 if x[3] == "J" else 1)

    # Drop rows where label mapping failed
    df.dropna(subset=["label_idx"], inplace=True)
    df["label_idx"] = df["label_idx"].astype(int)

    print(f"✅ Preprocessed {len(df)} rows. Clean text avg length: "
          f"{df['clean_text'].str.len().mean():.0f} chars")

    return df


# ── Stylometric features (extra signals) ────────────────────────────────────

def extract_stylometric_features(df: pd.DataFrame, text_col: str = "posts") -> pd.DataFrame:
    """
    Extract lightweight stylometric features from raw (not cleaned) posts:
        - avg_post_len      : average word count per post
        - exclamation_freq  : exclamation marks per 1000 chars
        - question_freq     : question marks per 1000 chars
        - caps_ratio        : ratio of uppercase letters
        - unique_word_ratio : vocabulary richness
    """
    def stylometrics(raw_text: str) -> dict:
        posts = raw_text.split("|||")
        lengths = [len(p.split()) for p in posts]
        total_chars = len(raw_text) + 1

        all_words  = raw_text.lower().split()
        unique     = set(all_words)

        return {
            "avg_post_len":      np.mean(lengths),
            "exclamation_freq":  raw_text.count("!") / total_chars * 1000,
            "question_freq":     raw_text.count("?") / total_chars * 1000,
            "caps_ratio":        sum(1 for c in raw_text if c.isupper()) / total_chars,
            "unique_word_ratio": len(unique) / (len(all_words) + 1),
        }

    feats = df[text_col].apply(stylometrics).apply(pd.Series)
    return feats


if __name__ == "__main__":
    # Quick smoke test
    sample = "I am an INFP. I love thinking about ideas!!! Check this out: https://example.com 😊"
    print("Raw   :", sample)
    print("Clean :", preprocess_single(sample))