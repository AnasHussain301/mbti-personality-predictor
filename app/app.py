import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_mbti


st.set_page_config(page_title="MBTI Personality Predictor", layout="centered")
st.title("MBTI Personality Predictor")

text = st.text_area(
    "Paste writing sample",
    height=220,
    placeholder="Paste a few paragraphs of natural writing for the best result.",
)

if st.button("Predict", type="primary"):
    if len(text.split()) < 20:
        st.warning("Please provide at least 20 words.")
    else:
        try:
            result = predict_mbti(text)
        except FileNotFoundError:
            st.error("Model files are missing. Run `python main.py --skip-eda` first.")
        else:
            st.metric("Predicted type", f"{result['mbti_type']} - {result['title']}", f"{result['confidence']:.1%}")
            st.write(result["description"])
            st.subheader("Top matches")
            for label, prob in result["top3"]:
                st.progress(prob, text=f"{label}: {prob:.1%}")
            st.subheader("Dimension confidence")
            for label, score in result["dimension_scores"].items():
                st.progress(score, text=f"{label}: {score:.1%}")
