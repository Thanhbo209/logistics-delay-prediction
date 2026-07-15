# Phase 2 – Model Training & Evaluation

## Objective

* Train a binary classification model to predict whether an inbound shipment will be
  highly delayed (delay exceeding 24 hours).
* Evaluate model performance honestly before treating the model as ready for use.
* Produce reproducible model artifacts for later inference and explainability.

## Model Selection

A Random Forest Classifier was selected for this prototype. Advantages relevant to
this problem:

* Handles the mix of numerical (queue length, port utilization, historical delay) and
  categorical (vessel type, weather) features well after simple encoding.
* Captures non-linear relationships and interactions between features — notably the
  compounding effect between queue length and port utilization — without manually
  engineering interaction terms.
* Requires minimal preprocessing: no feature scaling or normalization needed, unlike
  distance-based or linear models.
* Provides built-in feature importance, which the next phase (Explainability) builds on
  directly to satisfy the assignment's requirement to explain predictions.

Simpler alternatives (e.g. logistic regression) were considered but would require
manually specified interaction terms to capture effects the Random Forest learns natively.
More complex alternatives (e.g. gradient boosting, XGBoost) were not justified given the
dataset size (500 rows) and the 3-day prototype scope.

## Data Preparation

* Load the generated synthetic dataset (`src/data/port_data.csv`, 500 rows).
* Apply one-hot encoding to the categorical variables (`vessel_type`, `weather`).
* Separate features (`X`) from the target (`y = is_delayed`).
* Remove leakage columns from the feature set: `delay_hours` (the continuous value the
  target was thresholded from) and `is_delayed` (the target itself).
* Use a fixed random seed (`42`) throughout — data splitting, model fitting, and
  permutation-based steps — so results are reproducible across runs.

## Validation Strategy

* Stratified 5-Fold Cross-Validation is used to evaluate the model.
* Stratification preserves the ~37% delayed / ~63% on-time class distribution in every
  fold, rather than letting it vary by chance.
* ROC-AUC and F1 are computed per fold, then averaged.
* Out-of-fold predictions are also collected: every row is scored exactly once, by
  whichever fold treated it as held-out data. This produces one aggregate, honest
  evaluation across all 500 rows, distinct from the per-fold average.

With only 500 rows, a single train/test split would be high-variance — the reported
performance could shift substantially depending on which rows happened to land in the
test set. Cross-validation instead uses every row as test data exactly once, giving a
more stable and trustworthy performance estimate while preserving the class ratio.

## Final Model

After validation confirms the approach generalizes, a final Random Forest is trained on
the complete dataset (all 500 rows) — this is the model that ships. Cross-validation
models are used only to produce the evaluation metrics above and are discarded afterward.

Saved artifacts (`models/`):

* `model.joblib` — the trained classifier
* `feature_schema.json` — the exact feature column names and order the model expects,
  required so future inference code cannot silently misalign columns
* `metrics.json` — full cross-validation results (per-fold and out-of-fold)

## Results

| Metric | Value |
| --- | ---: |
| Mean ROC-AUC | 0.959 |
| Mean F1 | 0.843 |
| Out-of-Fold ROC-AUC | 0.958 |
| Out-of-Fold F1 | 0.844 |

Confusion matrix (out-of-fold predictions, threshold = 0.5):

| | Predicted No Delay | Predicted Delay |
| --- | ---: | ---: |
| Actual No Delay | 296 | 19 |
| Actual Delay | 36 | 149 |

## Observations

* Performance is consistent across all five folds, with no single fold diverging sharply
  from the others.
* Out-of-fold metrics closely match the fold-averaged metrics (0.958 vs. 0.959 ROC-AUC),
  indicating stable generalization rather than a result driven by one favorable split.
* The high ROC-AUC (~0.96) demonstrates strong ranking ability — the model reliably
  scores delayed shipments as higher-risk than on-time ones.
* The F1 score (~0.84) indicates a good balance between precision and recall, though the
  confusion matrix shows more false negatives (36) than false positives (19) — missed
  delays are currently the model's more common error type.
* The trained model and its saved artifacts are suitable inputs for the next phase,
  Explainability, without requiring any retraining.
