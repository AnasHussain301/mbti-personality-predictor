# MBTI Personality Predictor

Local training and inference pipeline for the Kaggle `mbti_1.csv` dataset.

## Run

```bash
cd "/mnt/d/New folder/personality_predictor"
/home/kingd/miniconda3/envs/mbti_env/bin/python main.py --skip-eda
```

Launch the app after training:

```bash
streamlit run app/app.py
```

The project saves model artifacts in `models/` and plots/results in `outputs/`.
