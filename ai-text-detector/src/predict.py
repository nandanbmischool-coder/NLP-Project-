import time
from functools import lru_cache
from pathlib import Path

import joblib

from src.features import clean_text, split_sentences

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "detector.joblib"
MIN_WORDS = 30
MIN_SENT_WORDS = 6
THRESH_AI = 0.65
THRESH_HUMAN = 0.35


@lru_cache(maxsize=1)
def _model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


def predict_text(text: str) -> dict:
    started = time.perf_counter()
    text = clean_text(text)
    word_count = len(text.split())
    if word_count < MIN_WORDS:
        return {"status": "too_short", "words": word_count, "min_words": MIN_WORDS}

    model = _model()
    probability = float(model.predict_proba([text])[0, 1])
    sentences = split_sentences(text)
    eligible = [sentence for sentence in sentences if len(sentence.split()) >= MIN_SENT_WORDS]
    probabilities = model.predict_proba(eligible)[:, 1] if eligible else []
    score_map = dict(zip(eligible, probabilities))
    sentence_scores = [{"sentence": sentence, "p_ai": float(score_map[sentence]) if sentence in score_map else None} for sentence in sentences]

    if probability >= THRESH_AI:
        label, verdict = "AI-generated", "ai"
    elif probability <= THRESH_HUMAN:
        label, verdict = "Human-written", "human"
    else:
        label, verdict = "Uncertain", "uncertain"

    return {
        "status": "ok",
        "p_ai": probability,
        "label": label,
        "verdict": verdict,
        "words": word_count,
        "sentence_count": len(sentences),
        "sentences": sentence_scores,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
