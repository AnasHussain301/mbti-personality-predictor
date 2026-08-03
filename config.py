"""
Shared configuration for the MBTI personality predictor project.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

# The CSV in this workspace currently lives in outputs/. Keep data/ as a
# fallback so either common layout works.
RAW_DATA_PATH = str(
    OUTPUTS_DIR / "mbti_1.csv"
    if (OUTPUTS_DIR / "mbti_1.csv").exists()
    else DATA_DIR / "mbti_1.csv"
)

MBTI_TYPES = [
    "ENFJ", "ENFP", "ENTJ", "ENTP",
    "ESFJ", "ESFP", "ESTJ", "ESTP",
    "INFJ", "INFP", "INTJ", "INTP",
    "ISFJ", "ISFP", "ISTJ", "ISTP",
]
TYPE_TO_IDX = {label: idx for idx, label in enumerate(MBTI_TYPES)}
IDX_TO_TYPE = {idx: label for label, idx in TYPE_TO_IDX.items()}

TEST_SIZE = 0.2
RANDOM_STATE = 42

TFIDF_MAX_FEATURES = 12000
SVD_COMPONENTS = 256
MIN_POST_LENGTH = 20

# ─── MLP Hyperparameters ────────────────────────────────────────────────────
# Reduced from (512, 256) to (256, 128) to prevent overfitting on 384-dim BERT
MLP_HIDDEN_LAYERS  = (256, 128)
MLP_DROPOUT        = 0.4
MLP_LEARNING_RATE  = 0.001
MLP_MAX_ITER       = 300
MLP_RANDOM_STATE   = 42

# ─── Training ───────────────────────────────────────────────────────────────
TEST_SIZE          = 0.2
RANDOM_STATE       = 42
CV_FOLDS           = 5

# ─── MBTI Descriptions ───────────────────────────────────────────────────────
MBTI_DESCRIPTIONS = {
    "INTJ": ("The Architect",     "Strategic, independent, and determined. INTJs thrive on planning and executing complex ideas."),
    "INTP": ("The Logician",      "Innovative thinkers driven by logic. INTPs love exploring abstract theories."),
    "ENTJ": ("The Commander",     "Bold and strong-willed natural leaders. ENTJs find a way where there isn't one."),
    "ENTP": ("The Debater",       "Smart and curious thinkers who love intellectual challenges and arguing both sides."),
    "INFJ": ("The Advocate",      "Quiet yet inspiring idealists with an innate desire to do what's right."),
    "INFP": ("The Mediator",      "Poetic, kind, and altruistic. INFPs are always searching for meaning and truth."),
    "ENFJ": ("The Protagonist",   "Charismatic and inspiring leaders. ENFJs are natural mentors and connectors."),
    "ENFP": ("The Campaigner",    "Enthusiastic and creative. ENFPs see life as full of possibilities and connections."),
    "ISTJ": ("The Logistician",   "Practical and fact-minded. ISTJs are reliable, driven by duty and responsibility."),
    "ISFJ": ("The Defender",      "Dedicated and warm protectors, always ready to defend those they care about."),
    "ESTJ": ("The Executive",     "Excellent administrators who value tradition, order, and getting things done."),
    "ESFJ": ("The Consul",        "Extraordinarily caring and social. ESFJs love helping and participating in community."),
    "ISTP": ("The Virtuoso",      "Bold and practical experimenters who master tools of all kinds."),
    "ISFP": ("The Adventurer",    "Flexible and charming. ISFPs live in the moment and love new experiences."),
    "ESTP": ("The Entrepreneur",  "Smart, energetic, and perceptive. ESTPs love living on the edge and taking action."),
    "ESFP": ("The Entertainer",   "Spontaneous and enthusiastic. ESFPs love being the center of attention."),
}

# ─── MBTI Descriptions ───────────────────────────────────────────────────────
MBTI_DESCRIPTIONS = {
    "INTJ": ("The Architect",     "Strategic, independent, and determined. INTJs thrive on planning and executing complex ideas."),
    "INTP": ("The Logician",      "Innovative thinkers driven by logic. INTPs love exploring abstract theories."),
    "ENTJ": ("The Commander",     "Bold and strong-willed natural leaders. ENTJs find a way where there isn't one."),
    "ENTP": ("The Debater",       "Smart and curious thinkers who love intellectual challenges and arguing both sides."),
    "INFJ": ("The Advocate",      "Quiet yet inspiring idealists with an innate desire to do what's right."),
    "INFP": ("The Mediator",      "Poetic, kind, and altruistic. INFPs are always searching for meaning and truth."),
    "ENFJ": ("The Protagonist",   "Charismatic and inspiring leaders. ENFJs are natural mentors and connectors."),
    "ENFP": ("The Campaigner",    "Enthusiastic and creative. ENFPs see life as full of possibilities and connections."),
    "ISTJ": ("The Logistician",   "Practical and fact-minded. ISTJs are reliable, driven by duty and responsibility."),
    "ISFJ": ("The Defender",      "Dedicated and warm protectors, always ready to defend those they care about."),
    "ESTJ": ("The Executive",     "Excellent administrators who value tradition, order, and getting things done."),
    "ESFJ": ("The Consul",        "Extraordinarily caring and social. ESFJs love helping and participating in community."),
    "ISTP": ("The Virtuoso",      "Bold and practical experimenters who master tools of all kinds."),
    "ISFP": ("The Adventurer",    "Flexible and charming. ISFPs live in the moment and love new experiences."),
    "ESTP": ("The Entrepreneur",  "Smart, energetic, and perceptive. ESTPs love living on the edge and taking action."),
    "ESFP": ("The Entertainer",   "Spontaneous and enthusiastic. ESFPs love being the center of attention."),
}
