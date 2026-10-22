import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "dataset.csv"
OUTPUT = BASE / "models" / "figure_4_3_wordclouds.svg"
SAMPLE_PER_CLASS = 5000


def read_balanced_sample() -> pd.DataFrame:
    chunks = []
    counts = {0: 0, 1: 0}
    for chunk in pd.read_csv(DATA, usecols=["text", "label"], chunksize=50_000):
        chunk = chunk.dropna(subset=["text", "label"])
        chunk["label"] = chunk["label"].astype(int)
        for label in (0, 1):
            available = chunk[chunk["label"] == label]
            remaining = SAMPLE_PER_CLASS - counts[label]
            if remaining > 0:
                selected = available.head(remaining)
                chunks.append(selected)
                counts[label] += len(selected)
        if all(value >= SAMPLE_PER_CLASS for value in counts.values()):
            break
    result = pd.concat(chunks, ignore_index=True)
    if any(value < SAMPLE_PER_CLASS for value in counts.values()):
        raise ValueError(f"Could not collect balanced samples: {counts}")
    return result


def svg_text(word: str, x: float, y: float, size: float, color: str, anchor: str) -> str:
    safe_word = html.escape(word)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size:.1f}px" '
        f'font-weight="700" fill="{color}">{safe_word}</text>'
    )


def cloud(words: list[tuple[str, float]], left: float, top: float, width: float, color: str) -> str:
    positions = [
        (0.50, 0.20), (0.28, 0.35), (0.72, 0.35), (0.50, 0.48),
        (0.20, 0.55), (0.80, 0.55), (0.36, 0.68), (0.64, 0.68),
        (0.50, 0.84), (0.15, 0.82), (0.85, 0.82), (0.30, 0.92),
        (0.70, 0.92), (0.12, 0.25), (0.88, 0.25),
    ]
    sizes = np.linspace(31, 13, len(words))
    parts = []
    for index, ((word, _score), (px, py)) in enumerate(zip(words, positions)):
        anchor = "middle"
        parts.append(svg_text(
            word,
            left + width * px,
            top + 65 + 185 * py,
            sizes[index],
            color,
            anchor,
        ))
    return "\n".join(parts)


def main() -> None:
    sample = read_balanced_sample()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_features=12_000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{2,}\b",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(sample["text"])
    names = np.array(vectorizer.get_feature_names_out())
    human = np.asarray(matrix[sample["label"].to_numpy() == 0].mean(axis=0)).ravel()
    ai = np.asarray(matrix[sample["label"].to_numpy() == 1].mean(axis=0)).ravel()

    def top_terms(scores: np.ndarray) -> list[tuple[str, float]]:
        adjusted = scores / (human + ai + 1e-9)
        indices = np.argsort(adjusted)[::-1][:15]
        return [(names[index], float(adjusted[index])) for index in indices]

    human_words = top_terms(human)
    ai_words = top_terms(ai)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="620" viewBox="0 0 1200 620">
  <rect width="1200" height="620" fill="#f7f8fc"/>
  <text x="600" y="42" text-anchor="middle" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#172033">Figure 4.3 — Class-specific influential terms</text>
  <rect x="35" y="72" width="545" height="510" rx="18" fill="#ffffff" stroke="#dfe4f2"/>
  <rect x="620" y="72" width="545" height="510" rx="18" fill="#ffffff" stroke="#dfe4f2"/>
  <text x="307" y="116" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#168342">Human-written texts</text>
  <text x="892" y="116" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#c4455d">AI-generated texts</text>
  {cloud(human_words, 55, 125, 505, "#328c62")}
  {cloud(ai_words, 640, 125, 505, "#c75b72")}
  <text x="600" y="610" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#718096">Terms ranked by class-specific mean TF-IDF influence • balanced sample of 5,000 texts per class</text>
</svg>
'''
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(svg, encoding="utf-8")
    print(f"Saved {OUTPUT}")
    print("Human terms:", ", ".join(word for word, _ in human_words[:3]))
    print("AI terms:", ", ".join(word for word, _ in ai_words[:3]))


if __name__ == "__main__":
    main()
