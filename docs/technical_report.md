# Shipment Delay Prediction Using Machine Learning

**Author:** Thanh\
**Date:** July 15, 2026\
**Purpose:** Technical report for the Safiri Ai AI Internship take-home assignment — a prototype system for predicting port congestion and shipment delays.

---

## 1. Introduction

Port congestion is one of the most significant sources of delay and uncertainty in global logistics. Vessel traffic, port capacity, weather, and operational inefficiencies interact in ways that cascade into missed delivery windows, higher costs, and reduced reliability for shippers and their customers. Anticipating these disruptions before they occur — rather than reacting to them — is what allows a logistics operation to plan around risk instead of absorbing it.

This project builds a prototype that estimates the likelihood of a shipment being significantly delayed, given its current operating conditions (vessel type, offshore queue length, port utilization, weather, and the route's recent delay history). The system does not stop at a probability: it also reports a confidence score and explains which factors drove each prediction, so the output is directly usable by a logistics manager rather than a black-box number. The objective was to demonstrate sound problem framing, a defensible modeling approach, and honest, interpretable evaluation — within a 3-day prototype scope, not a production system.

## 2. Dataset & Assumptions

The project uses a fully synthetic dataset of 500 shipments, generated with a documented, seeded formula so results are reproducible. Each row represents one inbound shipment and includes five input features — vessel type, weather conditions, queue length, port utilization, and historical average delay for the route — plus a binary target, `is_delayed`, defined as a simulated delay exceeding 24 hours. The dataset was intentionally designed to simulate realistic operational conditions (a minority-class delay rate of ~37%, mirroring the fact that congestion is the exception rather than the norm) while remaining fully reproducible and auditable. Full generation logic, feature definitions, and calibration rationale are documented in `dataset_design.md`.

Key assumptions, stated explicitly: the dataset is synthetic, standing in for real AIS or port-call data; a single, generic port is modeled with no per-port variation; each shipment is treated as an independent event; and all input features are known before the outcome, so the model has no access to leaked information.

## 3. Model Development

Categorical features (vessel type, weather) were one-hot encoded; no other preprocessing was required, since tree-based models do not need feature scaling. Two columns — `delay_hours` (the continuous value the target was thresholded from) and `is_delayed` itself — were explicitly excluded from the feature matrix to prevent label leakage.

A Random Forest Classifier was selected for this prototype because it handles the mix of numerical and categorical features with minimal preprocessing, captures non-linear interactions (notably between queue length and port utilization) without manually engineered interaction terms, and provides built-in feature importance that the explainability phase builds on directly. Simpler alternatives (e.g. logistic regression) would have required hand-built interaction terms; more complex alternatives (gradient boosting, XGBoost) were not justified for a 500-row prototype on a 3-day timeline.

The model was evaluated using stratified 5-fold cross-validation, which preserves the class ratio in every fold and gives a stable performance estimate on a dataset this size, where a single train/test split would be high-variance. ROC-AUC and F1 were used as the primary metrics rather than accuracy, since the classes are imbalanced and accuracy would reward a trivial "always predict on-time" model.

## 4. Results

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

The close agreement between the fold-averaged and out-of-fold metrics (0.959 vs. 0.958 ROC-AUC) indicates stable generalization rather than a result driven by a favorable split. The high ROC-AUC shows strong ranking ability — the model reliably scores delayed shipments as higher-risk than on-time ones — while the confusion matrix shows the model's more common error is missing a delay (36 false negatives) rather than raising a false alarm (19 false positives), a distinction worth weighing against the operational cost of each error type.

## 5. Model Explainability

Two independent methods were used to rank feature importance: the Random Forest's built-in impurity-based importance (`feature_importances_`), and permutation importance, which measures how much shuffling a feature degrades ROC-AUC. Permutation importance was used to drive the shipment-level explanations, not built-in importance, because impurity-based importance is known to overestimate continuous numerical features regardless of whether they are genuinely predictive.

This distinction was not theoretical — it was confirmed directly during implementation. `arrival_hour` ranked 4th of 10 features under built-in importance (a relatively high position), while permutation importance placed it 7th of 10 with a value of essentially zero. Since `arrival_hour` is documented as never having been wired into the underlying delay formula, this is a clear, demonstrated case of impurity-based importance's known bias. Relying on it for explanations would have meant citing a feature the model does not actually use as a reason for a prediction; the explainability implementation was corrected to filter on permutation importance specifically to avoid this.

Shipment-level explanations are generated from this corrected ranking: for a given shipment, only features the model demonstrably relies on, and whose value is notably different from the dataset average, are described — in plain language, without ML terminology.

## 6. Operational Insights

| Model Finding | Operational Response |
|---|---|
| High queue length | Increase berth allocation or rebalance vessel scheduling |
| High historical average delay | Notify customers earlier and monitor recurring congestion on that route |
| High port utilization | Delay arrivals, redistribute workload, or prepare additional resources |

The value of an interpretable prediction is that it does more than flag risk — it points at *why*, which is what turns a probability into a decision. A logistics manager reading "high delay risk, driven primarily by queue length well above normal" can act before the delay materializes: reallocating berths, adjusting scheduling, or proactively communicating with the customer, rather than discovering the delay only once it has already happened. This is the difference between a reactive and a proactive operation, and it is the core reason this project treats explainability as a first-class requirement alongside the prediction itself.

## 7. Limitations & Future Work

**Current limitations:**

- Trained and evaluated entirely on synthetic data, not real AIS or port-call records
- Only one model family (Random Forest) was evaluated; no comparative benchmarking
- Shipment-level explanations are rule-based and heuristic, not an exact mathematical attribution
- No deployment or API — the pipeline runs as sequential scripts
- No validation against external or real-world data

**Future work:**

- Evaluate Gradient Boosting / XGBoost as alternative model families
- Incorporate real operational or AIS data
- Implement SHAP-based local explanations for exact per-instance attribution
- Expose predictions through a REST API
- Package for production deployment

## 8. Conclusion

This project delivers a complete, reproducible prototype for predicting shipment delay risk: a documented synthetic dataset, a Random Forest classifier validated through stratified cross-validation (0.959 mean ROC-AUC, 0.843 mean F1), and an explainability layer that identifies which operating conditions drive each prediction — including catching and correcting a real bias in a naive importance-ranking approach along the way. The emphasis on interpretability throughout, rather than prediction accuracy alone, is what makes the system operationally useful: a logistics manager is given not just a risk score but a defensible reason for it, connected directly to concrete responses. Within the scope of a 3-day prototype, this demonstrates a sound, honestly-evaluated approach to a genuinely difficult logistics problem, with a clear and realistic path to production hardening.
