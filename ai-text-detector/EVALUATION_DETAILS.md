# DetectAI Evaluation Details

This file records the values currently supported by the project artifacts and
runtime measurements. Percentages below are calculated from the values in
`models/metrics.json`.

## Dataset and split

| Measure | Value |
|---|---:|
| Total texts (`N`) | 1,054,866 |
| Human texts | 700,000 |
| AI texts | 354,866 |
| Training texts (`N_train`) | 19,220 |
| Test texts | 4,780 |
| Dataset samples used for the reported evaluation | 24,000 |
| Split method | GroupShuffleSplit, test size 20%, random state 42 |

The source CSV contains 1,054,866 rows. The training script deliberately uses a
balanced reproducible subset capped at 12,000 texts per class before splitting,
which produces 24,000 evaluation samples in total.

## Model evaluation

| Metric | Value |
|---|---:|
| Accuracy | 97.8033% |
| Precision | 97.4678% |
| Recall | 98.1605% |
| F1-score | 97.8130% |
| ROC AUC | 99.8212% |
| False-positive rate | 2.5544% |
| False-negative rate | 1.8395% |

### Confusion matrix

| Actual \ Predicted | Human (0) | AI (1) |
|---|---:|---:|
| Human (0) | TN = 2,327 | FP = 61 |
| AI (1) | FN = 44 | TP = 2,348 |

These values are read directly from
[metrics.json](./models/metrics.json).

## Table 4.4 baseline comparison

The following four baselines were run on the same deterministic balanced
24,000-sample dataset and the same GroupShuffleSplit train/test split
(19,220/4,780, random state 42). The standalone word and character TF-IDF
baselines used a reproducibility cap of 50,000 features each to fit within the
available machine memory. Full results, including confusion counts, are saved
in [baseline_metrics.json](./models/baseline_metrics.json).

| Model | Accuracy | F1-score | ROC-AUC | FPR |
|---|---:|---:|---:|---:|
| Majority-class baseline | 49.81% | 0.665 | 0.500 | 100.0% |
| Stylometric features only | 71.23% | 0.705 | 0.784 | 26.7% |
| Word TF-IDF only | 88.63% | 0.889 | 0.957 | 13.9% |
| Character TF-IDF only | 90.08% | 0.903 | 0.961 | 12.5% |
| Proposed (word + character + stylometry) | 97.80% | 0.978 | 0.998 | 2.6% |

## Mean latency

Measured on the local warm model using 30 repeated calls to
`src.predict.predict_text` with the same 43-word input:

| Measure | Value |
|---|---:|
| Mean | 12.78 ms |
| Median | 12.17 ms |
| Minimum | 8.53 ms |
| Maximum | 24.59 ms |

This is model inference latency only. It excludes browser rendering and network
round-trip time.

## Environment

| Component | Detail |
|---|---|
| OS | Microsoft Windows 11 Home Single Language, 64-bit, version 10.0.26200 |
| CPU | 12th Gen Intel(R) Core(TM) i5-12450H |
| CPU cores / logical processors | 8 / 12 |
| RAM | 15.71 GB |
| System disk | 475.67 GB total, 71.63 GB free at measurement time |
| Python | 3.11.9 |
| IDE | Visual Studio Code |
| Frontend | React with Vite |
| Backend | FastAPI with Uvicorn |

## Figures

### Figure 4.1 — Confusion matrix

Use the confusion-matrix table above to create the figure directly from the
verified counts: `[[2327, 61], [44, 2348]]`, with rows as actual labels
`[Human, AI]` and columns as predicted labels `[Human, AI]`.

### Figure 4.2 — ROC curve

The current `metrics.json` stores only the aggregate ROC AUC
(`0.9982116897195005`), not the per-example test labels and predicted
probabilities required to reconstruct the ROC curve. A ROC image cannot be
reconstructed faithfully from the aggregate AUC alone. Regenerate this figure
by updating the training evaluation to save `test["label"]` and
`predict_proba(test["text"])[:, 1]`, then pass those arrays to
`sklearn.metrics.roc_curve`.

### Figure 4.3 — Word clouds

The generated figure is available at
[figure_4_3_wordclouds.svg](./models/figure_4_3_wordclouds.svg). It uses a
balanced sample of 5,000 human texts and 5,000 AI texts. Terms are ranked by
class-specific mean TF-IDF influence after English stop-word removal. In this
sample, the most prominent human-side terms include **essay**, **table**, and
**contents**; the AI-side terms include **it's**, **don't**, and **you're**.
These are corpus-level signals, not authorship proof.

### Figure 4.4 — Web interface

The working interface is available at:

**http://127.0.0.1:5173/**

The backend API documentation is available at:

**http://127.0.0.1:8000/docs**
