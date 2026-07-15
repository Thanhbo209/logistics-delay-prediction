# Phase 3 – Model Explainability

## Objective

Phase 2 produced a classifier that predicts delay likelihood with a confidence score,
but a probability alone does not tell a logistics manager *why* a shipment was flagged.
This phase adds that missing piece: it explains, in plain language, which factors drove
each prediction — satisfying the assignment's third requirement ("explain the key
factors contributing to the prediction") without retraining or otherwise modifying the
Phase 2 model.

## Implementation

`src/explain_model.py` loads the trained model and feature schema from Phase 2 and
produces two layers of explanation:

* **Global feature importance**, computed two ways: the Random Forest's built-in
  impurity-based importance (`feature_importances_`), and permutation importance
  (`sklearn.inspection.permutation_importance`, 10 repeats, seeded).
* **Per-shipment explanations**, generated for two illustrative examples — the
  shipment with the highest predicted delay probability and the one with the lowest,
  chosen deterministically rather than at random so the pair is reproducible and
  contrastive.

Workflow: load model and schema → rebuild the feature matrix using the same encoding
function as training (`train_model.build_features`, imported rather than duplicated) →
compute both importance rankings → save them → compute per-shipment predictions and
translate the model's top demonstrated drivers into manager-facing sentences, using
dataset-wide statistics (mean, median, standard deviation) only as a reference point for
phrasing, never as model input.

A key implementation choice: explanations are built from the **permutation** ranking,
not the built-in one. Only features whose permutation importance clears a configurable
threshold (`MIN_IMPORTANCE = 0.01`) are eligible for mention, and a feature is only
called out for a given shipment if its value is at least 0.5 standard deviations from
the dataset mean — trivially-typical values aren't narrated even if the feature is
generally important.

## Design Decisions

* **Permutation importance over built-in importance for explanations.** Built-in
  (impurity-based) importance is known to overestimate continuous/high-cardinality
  features regardless of real predictive value. This was confirmed directly during
  implementation (see Challenges) and is the reason explanations are driven by
  permutation importance instead, with built-in importance retained only as a
  side-by-side comparison in the global summary.
* **No SHAP or third-party attribution libraries.** SHAP has no prebuilt wheel for this
  project's environment (Windows, Python 3.14) and would require a source build —
  disproportionate risk for a 3-day prototype. `feature_importances_` and
  `permutation_importance` are both native to scikit-learn, already a project
  dependency, and sufficient to satisfy the assignment's explainability requirement.
* **Rule-based, template-driven language generation over free-form text generation.**
  A fixed set of phrase templates keyed by feature and direction (above/below typical)
  keeps explanations deterministic, auditable, and dependency-free, at the cost of being
  less flexible than a generative approach — an acceptable trade-off given the small,
  fixed feature set.
* **Permutation importance evaluated on the full dataset with the final model**, not
  averaged across cross-validation folds. This is technically training-data evaluation,
  but is treated as acceptable here because it produces a relative *ranking* of
  features, not a claimed generalization performance number (unlike Phase 2's confusion
  matrix, which specifically used out-of-fold predictions because it does claim a
  performance number).
* **Reproducibility**: `random_state=42` on both the trained model (inherited from
  Phase 2's artifact) and the permutation importance computation. Verified by running
  the script twice and diffing `feature_importance.json` — byte-identical.

## Outputs

| Artifact | Purpose |
|---|---|
| `models/feature_importance.json` | Both importance rankings (`built_in`, `permutation`), sorted descending, rounded to 6 decimals — the source data for report tables/figures |
| Console output (global ranking) | Human-readable comparison of both importance methods, every feature |
| Console output (2 shipment explanations) | Worked examples of the plain-language explanation format, one high-risk and one low-risk shipment |
| Console output (sanity check) | Automated cross-check confirming `arrival_hour` — documented in `dataset_design.md` as carrying no real signal — is correctly identified as low-importance by permutation importance, despite ranking artificially higher under built-in importance |

No model artifacts are modified or re-saved in this phase; `models/model.joblib` and
`models/feature_schema.json` remain exactly as Phase 2 produced them.

## Results

Global importance (top 3 of 10 features, full ranking in `feature_importance.json`):

| Feature | Built-in Importance | Permutation Importance |
|---|---:|---:|
| queue_length | 0.315 | 0.162 |
| historical_avg_delay | 0.290 | 0.117 |
| port_utilization | 0.141 | 0.015 |

Both methods agree on the same top 3 drivers and their relative order. Only these three
features clear the `MIN_IMPORTANCE` threshold used for explanations — `weather` and
`vessel_type` contribute measurably to the synthetic ground-truth formula but not
enough, in this trained model, to pass the bar for being cited as a reason.

Example output (highest-risk shipment in the dataset):

> **High Delay Risk (100%)**
> Queue length is significantly higher than normal, indicating heavy congestion at the
> port. This is the single biggest factor in this prediction. This route has a recent
> history of longer-than-average delays. The port is operating close to full capacity,
> making delays more likely.

## Challenges

The most significant finding was a genuine disagreement between the two importance
methods on `arrival_hour`: built-in importance ranked it 4th of 10 (0.077, inside the
top half), while permutation importance ranked it 7th of 10 with a value of
essentially 0.000. Since `arrival_hour` is documented as carrying no real signal by
design, this was a concrete, demonstrated case of built-in importance's known bias
toward continuous numerical features. The initial implementation used built-in
importance to select explanation features and was therefore citing `arrival_hour` in a
shipment's explanation despite it being noise — a correctness issue, not a style issue.
Resolved by switching explanation feature-selection to permutation importance and
adding an automated sanity check that specifically compares the two methods on this
feature and explains any discrepancy, rather than silently reporting a rank number.

## Limitations

* Explanations are template-based, not a mathematically exact per-instance decomposition
  (as SHAP or `treeinterpreter` would provide) — they describe *which* known drivers are
  elevated for a shipment, not each feature's precise numeric contribution to the
  predicted probability.
* Permutation importance is computed on the full dataset rather than averaged across
  cross-validation folds; acceptable for a ranking, as noted in Design Decisions, but
  a more rigorous version would match Phase 2's out-of-fold approach.
* Only two example shipments are demonstrated in console output; the underlying
  functions generalize to any shipment, but a full batch-explanation report was out of
  scope for this phase.
* A single, generic port and route context is assumed, consistent with Phase 1's
  documented assumptions — explanations do not account for port-specific or
  route-specific baselines.

## Next Phase

With prediction, confidence, and explanation all implemented and validated, all three
core requirements from the assignment brief are now satisfied end-to-end. The next
phase consolidates this work into the assignment's required deliverables: the technical
report and an optional demo, drawing directly on the metrics (Phase 2) and importance
rankings/explanations (this phase) already produced and saved to `models/`.
