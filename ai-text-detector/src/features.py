import re

import numpy as np
from nltk.tokenize import wordpunct_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def clean_text(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def split_sentences(text: str):
    return [sentence for sentence in SENT_SPLIT.split(clean_text(text)) if sentence]


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Stylometric features: length, lexical diversity, burstiness, punctuation."""

    def fit(self, X, y=None):
        del X, y
        return self

    def transform(self, X):
        rows = []
        for text in X:
            text = clean_text(text)
            tokens = wordpunct_tokenize(text.lower())
            words = [word for word in tokens if word.isalpha()]
            n_words = max(len(words), 1)
            sentence_lengths = [len(wordpunct_tokenize(sentence)) for sentence in split_sentences(text)] or [0]
            rows.append([
                np.log1p(len(text)),
                np.log1p(n_words),
                len(set(words)) / n_words,
                float(np.mean(sentence_lengths)),
                float(np.std(sentence_lengths)),
                sum(text.count(char) for char in ",;:") / n_words,
                float(np.mean([len(word) for word in words])) if words else 0.0,
            ])
        return np.asarray(rows, dtype=float)


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word_tfidf", Pipeline([("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True
        ))])),
        ("char_tfidf", Pipeline([("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3,
            max_features=300000, sublinear_tf=True
        ))])),
        ("style", Pipeline([("f", StyleFeatures()), ("scale", StandardScaler())])),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(
            C=4.0, solver="liblinear", class_weight="balanced", max_iter=1000
        )),
    ])
