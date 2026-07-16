# Project Demonstration

This document walks through the complete machine learning workflow — from data generation to prediction explanation — as a written substitute for a live demo.

## Workflow Overview

```text
Generate Dataset
        ↓
Train & Evaluate Model
        ↓
Explain Predictions
```

- **Generate Dataset** — simulates synthetic shipment data with a documented delay formula.
- **Train & Evaluate Model** — fits a Random Forest classifier and validates it with cross-validation.
- **Explain Predictions** — computes global and shipment-level explanations from the trained model.

## Step 1 — Generate Dataset

```bash
python src/generate_data.py
```

Simulates 500 inbound shipments (vessel type, weather, queue length, port utilization, historical delay) and derives each shipment's delay outcome from a documented formula. Output is written to `src/data/port_data.csv`. The run prints a summary of the delay rate and feature distributions so the dataset can be sanity-checked immediately.

## Step 2 — Train the Model

```bash
python src/train_model.py
```

Encodes categorical features, removes leakage columns, and evaluates a Random Forest classifier using Stratified 5-Fold Cross Validation — preserving the delay/no-delay ratio in every fold and producing honest, out-of-fold performance estimates. The final model is refit on the full dataset and saved.

```text
Mean ROC-AUC: 0.959
Mean F1:      0.843
Out-of-fold ROC-AUC: 0.958
Out-of-fold F1:      0.844
```

## Step 3 — Explain the Model

```bash
python src/explain_model.py
```

Loads the trained model (no retraining) and produces global feature importance from two independent methods, plain-language explanations for individual shipments, and a sanity check confirming that a known non-predictive feature (`arrival_hour`) is correctly identified as low-importance.

```text
Permutation Importance
1. queue_length
2. historical_avg_delay
3. port_utilization

Shipment #105 — High Delay Risk (100%)
* Queue length is significantly higher than normal, indicating heavy congestion at the port.
* This route has a recent history of longer-than-average delays.
* The port is operating close to full capacity, making delays more likely.
```

## Generated Artifacts

| Artifact | Description |
|---|---|
| `model.joblib` | Trained Random Forest model |
| `metrics.json` | Cross-validation metrics |
| `feature_schema.json` | Feature ordering for inference |
| `feature_importance.json` | Explainability results |

## End-to-End Pipeline Summary

1. Generate a synthetic logistics dataset.
2. Train and evaluate a Random Forest classifier.
3. Save the trained model.
4. Compute feature importance.
5. Explain individual shipment predictions.

## Key Takeaways

- End-to-end, reproducible ML pipeline — every stage is seeded and deterministic.
- Strong predictive performance (0.959 mean ROC-AUC).
- Explainable predictions suitable for operational decision support.
- Lightweight implementation using only the Python scientific ecosystem.

## Notes

No graphical user interface was developed — this project focuses on the machine learning workflow rather than frontend presentation. The command-line pipeline, together with this walkthrough and the accompanying documentation, serves as the project demonstration.
