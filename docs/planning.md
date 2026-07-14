# Planning — Port Congestion / Shipment Delay Prediction

*Safiri Ai 3-day take-home. Locked spec; becomes Sections 1–2 of the final report.
For dataset generation details, see `dataset_design.md`.*

## 1. Problem statement

Given an inbound shipment's operating conditions (vessel type, offshore queue, port
utilization, weather, route's recent delay history), predict **how likely it is to be
highly delayed**, report a **confidence score**, and **explain the driving factors** —
so a logistics manager can act early (notify the customer, reschedule labor, reroute cargo).

## 2. Prediction target

- **Framing:** binary classification, `is_delayed ∈ {0, 1}`.
- **"Delayed" means:** simulated delay exceeds **24 hours**. Chosen as the point where
  cascading costs (storage, re-planning, SLA penalties) become material — an operational
  judgment call, not derived from data.
- **Ground truth:** a continuous `delay_hours` is simulated from a documented formula
  (see `dataset_design.md`), then thresholded at 24h. Keeping `delay_hours` allows a
  future regression view without regenerating data.
- **Class balance target:** delayed should be a realistic **minority (~30–45%)** —
  congestion is the exception, not the norm. Currently **37%**.

## 3. Confidence score

The model's `predict_proba` output — a probability in [0, 1], not a raw score. Reported
as-is; calibration is a possible future step if predicted probabilities prove unreliable.

## 4. Evaluation

- **Primary:** ROC-AUC and F1 (positive/delayed class).
- **Secondary:** confusion matrix at the 0.5 threshold, for interpretability.
- **Method:** stratified 5-fold cross-validation. With 500 rows, a single train/test
  split is high-variance; CV gives an honest estimate and preserves class ratio per fold.
- **Why not accuracy:** classes are imbalanced (37/63), so "always predict on-time"
  would score well on accuracy while catching zero delays. AUC/F1 measure actual
  discrimination.

## 5. Assumptions

- Dataset is synthetic (permitted by the brief); generation is documented and seeded.
- All features are known before the outcome — no leakage.
- Single, generic port modeled; per-port variation is future work.
- Each shipment is treated as independent.
- Ground truth follows the simulated formula; real operations have unmodeled factors
  (labor actions, customs, mechanical failures) — acknowledged as a limitation.

## 6. Model choice

Tree-based model (Random Forest), not a black box — the brief requires explaining "why."

**Explainability: `feature_importances_` + permutation importance, not SHAP.** The brief
allows either. SHAP has no prebuilt wheel for this project's environment (Windows,
Python 3.14) and would need a source build — unjustified risk for a 3-day prototype.
sklearn's built-in tools give faithful, dependency-free explanations. SHAP noted as a
future enhancement if run on Linux/macOS or Python ≤3.12.

---

### Brief → decision traceability

| Brief requirement | Satisfied by |
|---|---|
| Predict congestion/delay likelihood | §2 binary classifier |
| Confidence score | §3 predicted probability |
| Explain why | §6 tree model + feature/permutation importance |
| Actionability | Top drivers mapped to recommendations (later phase) |
