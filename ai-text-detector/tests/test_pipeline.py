import numpy as np

from src.features import StyleFeatures, build_pipeline, clean_text, split_sentences

HUMAN = "honestly i dunno it was kinda weird we got lost twice lol my phone died halfway through"
AI = "In conclusion it is important to note that this comprehensive approach offers several advantages"


def test_clean_and_split():
    assert clean_text("  a   b \n c ") == "a b c"
    assert len(split_sentences("Hello there. How are you? Fine!")) == 3


def test_style_features_shape():
    assert StyleFeatures().transform(["One sentence here.", ""]).shape == (2, 7)


def test_pipeline_trains_and_predicts():
    texts = [f"{HUMAN} number {i}." for i in range(30)] + [f"{AI} number {i}." for i in range(30)]
    labels = [0] * 30 + [1] * 30
    model = build_pipeline().fit(texts, labels, clf__sample_weight=np.ones(60))
    probabilities = model.predict_proba([f"{AI} number 99."])[0]
    assert abs(probabilities.sum() - 1) < 1e-6
    assert probabilities[1] > 0.5
