# Planning — Port Congestion / Shipment Delay Prediction

*Safiri Ai 3-day take-home. This note is the locked specification for the project and
becomes Sections 1–2 of the final report.*

## 1. Problem statement

Port congestion causes shipment delays that cascade into missed delivery windows, higher
costs, and lost reliability. This project builds a prototype that, given the current
operating conditions for an inbound shipment (vessel type, offshore queue, port
utilization, weather, and the route's recent delay history), estimates **how likely that
shipment is to be highly delayed**, reports a **confidence score**, and **explains the
factors driving the prediction** — so a logistics manager can act early (notify the
customer, reschedule labor, or reroute cargo).

## 2. Prediction target (what "1" means)

- **Framing:** binary classification. `delay_flag ∈ {1 = highly delayed, 0 = not}`.
- **Definition of "highly delayed":** the shipment's delay exceeds **24 hours** beyond
  schedule. A full extra day is the point where cascading supply-chain costs (storage,
  re-planning, SLA penalties) become material, so it is a meaningful operational boundary.
- **How ground truth is produced (synthetic):** we simulate a continuous `delay_hours`
  from a documented formula over the features plus small Gaussian noise, then apply the
  24h threshold to get `delay_flag`. Keeping the latent `delay_hours` lets us later add a
  regression view without changing the data.
- **Class balance target:** positives (delayed) should be a realistic **minority (~30–45%)**
  — not 50/50, not 5/95. We tune the threshold/noise to land in this band.

## 3. Confidence score

The confidence score is the model's **predicted probability of delay** (`predict_proba`),
a value in [0, 1]. We will optionally calibrate it later so that "0.8" genuinely means
"~80% of such shipments are delayed." We report the probability, never a raw, unbounded
score.

## 4. Success metrics & evaluation

- **Primary:** ROC-AUC (ranking quality) and F1 on the positive/delayed class.
- **Secondary:** confusion matrix at the 0.5 threshold for interpretability.
- **Method:** **stratified k-fold cross-validation** (5-fold). With only 300 rows a single
  train/test split is high-variance; CV gives an honest estimate and preserves class ratio.
- **Why not accuracy:** the classes are imbalanced, so accuracy would reward "predict
  on-time always." AUC and F1 reflect the ability to actually catch delays.

## 5. Assumptions (explicitly stated)

- The dataset is **synthetic** (permitted by the brief); its generation logic is fully
  documented and seeded for reproducibility.
- All features are **known at prediction time** (before the outcome), so there is no leakage.
- A single, generic port is modeled; per-port variation is noted as future work.
- Each shipment outcome is treated as **independent**.
- Ground truth follows the simulated formula in §2; real operations would add unmodeled
  factors (labor actions, customs, mechanical failures), acknowledged as a limitation.

## 6. Decision that echoes forward

Because the brief requires **explaining "why,"** we commit now to an **explainable,
tree-based model** (Random Forest / gradient-boosted trees) rather than a black box.

**Explainability method — feature importances + permutation importance (not SHAP).**
The brief allows *"feature importances or SHAP"* and marks SHAP optional. We deliberately
skip SHAP: on the project's environment (Windows + Python 3.14) `shap` has no prebuilt
wheel and would require a source build, an unjustified risk for a 3-day prototype. A
tree model's built-in `feature_importances_` (global) plus scikit-learn
`permutation_importance` (global, model-agnostic) and per-prediction driver analysis give
us faithful, defensible explanations with zero dependency risk. SHAP is noted as a
future enhancement (best run on Linux/macOS or Python ≤3.12).

---

### Brief → decision traceability

| Brief requirement | Satisfied by |
|---|---|
| Predict congestion/delay likelihood | §2 binary classifier |
| Confidence score | §3 predicted probability |
| Explain why | §6 tree model + feature/permutation importance (SHAP skipped — see §6) |
| Actionability | Recommendations mapped from top drivers (later phase) |
