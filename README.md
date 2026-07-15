# Logistic Delay Prediction

Predicting shipment delays using a Random Forest classifier with model explainability for logistics decision support.

## Project Overview

Port congestion is one of the most significant sources of delay and uncertainty in global logistics. Vessel traffic, port capacity, weather, and operational inefficiencies interact to produce delays that cascade into missed delivery windows, higher costs, and reduced reliability for shippers and customers.

This prototype estimates the likelihood that an inbound shipment will be significantly delayed, given its current operating conditions (vessel type, queue length, port utilization, weather, and the route's recent delay history). It reports a confidence score rather than a bare label, and explains which factors drove each prediction — so the output is something a logistics manager can act on, not just a number.

The project demonstrates an end-to-end, reproducible ML workflow on a documented synthetic dataset: data generation, cross-validated model training, and explainability, built to a 3-day take-home scope rather than as a production system.

## Features

- Synthetic logistics dataset generation, seeded and reproducible
- Random Forest delay prediction with a probability-based confidence score
- Stratified 5-Fold Cross Validation
- Model evaluation (ROC-AUC, F1, confusion matrix), including honest out-of-fold metrics
- Global feature importance via two independent methods (built-in and permutation)
- Shipment-level, plain-language explanations
- Fully reproducible pipeline (fixed random seeds throughout)

## Project Architecture

```text
Generate Dataset
        ↓
Train Model
        ↓
Cross Validation
        ↓
Save Model
        ↓
Explain Predictions
```

- **Generate Dataset** — simulates 500 shipments with a documented delay formula and saves them to CSV.
- **Train Model** — encodes features and fits a Random Forest classifier.
- **Cross Validation** — stratified 5-fold CV produces honest, held-out performance estimates before anything is deployed.
- **Save Model** — the final model (refit on all data), its feature schema, and CV metrics are persisted to `models/`.
- **Explain Predictions** — loads the saved model (no retraining) and produces global and per-shipment explanations.

## Project Structure

```text
logistic-delay-prediction/
├── docs/
│   ├── planning.md                    # Problem framing, target definition, evaluation plan
│   ├── dataset_design.md              # Dataset schema and delay-simulation formula
│   ├── model_training.md              # Phase 2 summary: model choice, validation, results
│   ├── model_explainability.md        # Phase 3 summary: explainability approach and findings
│   └── safiri_take_home_problem_6.pdf # Original assignment brief
├── models/
│   ├── model.joblib                   # Trained Random Forest classifier
│   ├── feature_schema.json            # Exact feature column order the model expects
│   ├── metrics.json                   # Cross-validation results
│   └── feature_importance.json        # Built-in and permutation importance rankings
├── src/
│   ├── generate_data.py               # Synthetic dataset generator
│   ├── train_model.py                 # Training and cross-validation
│   ├── explain_model.py               # Explainability (loads the trained model, no retraining)
│   └── data/
│       └── port_data.csv              # Generated dataset (reproducible via generate_data.py)
├── pyproject.toml                     # Project dependencies
├── uv.lock                            # Locked dependency versions
└── README.md
```

## Tech Stack

| Technology | Purpose |
|---|---|
| Python (≥3.14) | Core implementation language |
| pandas | Dataset generation, CSV I/O, one-hot encoding |
| NumPy | Numerical operations (noise simulation, array math) |
| scikit-learn | Random Forest, stratified cross-validation, metrics, permutation importance |
| joblib | Serializing and loading the trained model |
| uv | Dependency locking and environment management |

## Installation

Requires Python 3.14+.

```bash
# 1. Clone the repository
git clone <repository-url>
cd logistic-delay-prediction

# 2. Install dependencies (creates/updates the project's virtual environment)
uv sync
```

If `uv` isn't available, install the same dependencies into a standard virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install pandas numpy scikit-learn joblib
```

## Running the Project

Run in order from the project root:

```bash
python src/generate_data.py
```
Generates a synthetic 500-row shipment dataset and writes it to `src/data/port_data.csv`.

```bash
python src/train_model.py
```
Encodes features, runs stratified 5-fold cross-validation, fits the final model on the full dataset, and saves `models/model.joblib`, `models/feature_schema.json`, and `models/metrics.json`.

```bash
python src/explain_model.py
```
Loads the trained model (no retraining), computes global feature importance, prints two worked shipment explanations, and saves `models/feature_importance.json`.

## Model Performance

Stratified 5-fold cross-validation results:

| Metric | Value |
|---|---:|
| Mean ROC-AUC | 0.959 |
| Mean F1 | 0.843 |
| Out-of-Fold ROC-AUC | 0.958 |
| Out-of-Fold F1 | 0.844 |

Confusion matrix (out-of-fold predictions, threshold = 0.5):

| | Predicted No Delay | Predicted Delay |
|---|---:|---:|
| Actual No Delay | 296 | 19 |
| Actual Delay | 36 | 149 |

Out-of-fold metrics closely match the fold-averaged metrics, indicating stable generalization rather than a result driven by one favorable data split.

## Explainability

Two independent methods are used to rank feature importance:

- **Built-in importance** (`feature_importances_`) — fast, native to the Random Forest, but known to overestimate continuous numerical features regardless of whether they're actually predictive.
- **Permutation importance** (`sklearn.inspection.permutation_importance`) — measures how much shuffling a feature degrades ROC-AUC, a more reliable indicator of real predictive reliance.

Shipment-level explanations are generated from **permutation** importance, not built-in importance. This project found a concrete case where the two disagree (see Key Findings), and citing built-in importance would have meant explaining a prediction using a feature the model doesn't actually rely on. Only features that clear a minimum importance threshold, and whose value is meaningfully different from the dataset average for that shipment, are mentioned — explanations are plain-language and avoid raw ML terminology.

## Key Findings

- **Queue length is the strongest predictor** of delay (permutation importance 0.162, by far the highest).
- **Historical average delay is highly influential** (permutation importance 0.117), reflecting route-level delay patterns.
- **Port utilization contributes to congestion risk**, though more modestly than queue length or delay history.
- **Arrival hour carries little real predictive signal**, despite ranking artificially higher (4th of 10) under built-in importance — permutation importance correctly places it near the bottom (7th of 10, ≈0.000), consistent with it never being wired into the synthetic delay formula.

## Assumptions

- Dataset is synthetic; generation logic is documented and seeded for reproducibility.
- A single, generic port is modeled — no port-specific or route-specific variation.
- Each shipment is treated as an independent event.
- All features are known before the outcome (no data leakage).
- The operating environment is simplified relative to real port operations.

## Limitations

- Trained and evaluated entirely on synthetic data, not real AIS or port-call records.
- Only one model family (Random Forest) is evaluated; no comparison against alternatives.
- Shipment-level explanations are rule-based and heuristic, not an exact mathematical attribution (e.g. SHAP).
- No real-time inference interface — the pipeline runs as sequential scripts, not a service.
- Not deployed or packaged for production use.

## Future Improvements

- Evaluate additional model families (gradient boosting, logistic regression baseline) for comparison.
- Replace or augment synthetic data with real operational or AIS data.
- Implement SHAP-based local explanations for exact per-instance attribution.
- Expose predictions through a REST API.
- Package and deploy as a small web application for interactive use.

## Documentation

- [`docs/planning.md`](docs/planning.md) — problem framing, target definition, and evaluation plan
- [`docs/dataset_design.md`](docs/dataset_design.md) — dataset schema and delay-simulation formula
- [`docs/model_training.md`](docs/model_training.md) — model selection, validation strategy, and results
- [`docs/model_explainability.md`](docs/model_explainability.md) — explainability approach and findings
- [`docs/safiri_take_home_problem_6.pdf`](docs/safiri_take_home_problem_6.pdf) — original assignment brief

## License

This repository was developed as a technical assessment / educational project and is not licensed for production or commercial use.
