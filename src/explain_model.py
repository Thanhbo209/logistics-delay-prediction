"""Phase 3 - Model Explainability.

Loads the Random Forest trained in Phase 2 (src/train_model.py) and explains it:
global feature importance (two methods) and plain-language, per-shipment reasoning.
This script never calls .fit() - it only loads and inspects an already-trained model.

Explanations are driven by PERMUTATION importance, not built-in impurity importance.
Built-in importance is known to overestimate continuous numerical features (see
_check_signal_agreement's docstring) - using it to decide what to tell a logistics
manager would risk citing a feature the model doesn't actually rely on.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from train_model import DATA_PATH, MODEL_DIR, build_features

MODEL_PATH = MODEL_DIR / "model.joblib"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"

RANDOM_SEED = 42
N_REPEATS = 10

# Raw numeric columns (pre-encoding) used to phrase per-shipment explanations.
# The one-hot dummy columns are 0/1 flags, not meaningful for mean/median comparison.
NUMERIC_FEATURES = ["queue_length", "port_utilization", "historical_avg_delay", "arrival_hour"]

# A feature must clear this permutation importance to be mentioned in an explanation
# at all. Below this, the model doesn't meaningfully rely on it - claiming otherwise
# would mislead the reader. Configurable: raise it to get terser, higher-confidence
# explanations; lower it to surface weaker signals.
MIN_IMPORTANCE = 0.01

# At most this many of the top permutation-ranked features (that also clear
# MIN_IMPORTANCE) are described per shipment, so an explanation never turns into a
# dump of every column.
MAX_FEATURES_PER_EXPLANATION = 4

# A numeric feature is only called out as "elevated"/"reduced" if it is at least this
# many standard deviations from the dataset mean - a small, data-driven bar rather than
# an arbitrary percentage, so trivially-close-to-average values aren't over-narrated.
SIGNIFICANCE_Z_THRESHOLD = 0.5

# Thresholds for the built-in-vs-permutation agreement check (see
# _check_signal_agreement). A feature counts as "relatively high" in built-in
# importance above this value, and "near zero" in permutation importance below it.
BUILTIN_RELATIVELY_HIGH = 0.05
PERMUTATION_NEAR_ZERO = MIN_IMPORTANCE

# arrival_hour is documented in docs/dataset_design.md as generated but never wired
# into the delay formula - it should carry no real signal. It's the one feature we
# know the ground-truth answer for, so it doubles as a built-in sanity check on the
# importance methods themselves.
SANITY_CHECK_FEATURE = "arrival_hour"

# Manager-facing phrases for numeric features, keyed by whether the shipment's value
# is significantly above or below the dataset's typical value. Deliberately avoids ML
# terminology ("importance", "coefficient") and raw numbers in favor of plain language.
NUMERIC_PHRASES_HIGH = {
    "queue_length": (
        "Queue length is significantly higher than normal, indicating heavy congestion at the port."
    ),
    "port_utilization": (
        "The port is operating close to full capacity, making delays more likely."
    ),
    "historical_avg_delay": "This route has a recent history of longer-than-average delays.",
    "arrival_hour": "This shipment is arriving later in the day than is typical for this route.",
}
NUMERIC_PHRASES_LOW = {
    "queue_length": "Queue length is lower than normal, indicating a smoothly flowing port.",
    "port_utilization": (
        "The port has considerable spare capacity right now, reducing the likelihood of delay."
    ),
    "historical_avg_delay": "This route has a strong recent on-time track record.",
    "arrival_hour": "This shipment is arriving earlier in the day than is typical for this route.",
}

# Manager-facing phrases for one-hot categorical features, keyed by (raw column, category).
CATEGORICAL_PHRASES = {
    ("weather", "Storm"): "Storm conditions are present at arrival, a known driver of delay.",
    ("weather", "Rain"): "Rainy conditions are present at arrival, a moderate driver of delay.",
    ("weather", "Clear"): "Weather conditions are clear, which typically supports on-time arrival.",
    ("vessel_type", "Tanker"): (
        "This is a Tanker, a vessel class that historically takes longer to handle."
    ),
    ("vessel_type", "Bulk"): (
        "This is a Bulk carrier, a vessel class with moderately slower handling."
    ),
    ("vessel_type", "Container"): (
        "This is a Container vessel, typically the fastest vessel class to handle."
    ),
}


def load_model() -> tuple[RandomForestClassifier, list[str]]:
    """Load the Phase 2 model and its feature schema. Does not retrain.

    Raises FileNotFoundError with the original message if either artifact is
    missing - that's a real precondition failure (Phase 2 hasn't been run yet),
    not something explanation logic should paper over.
    """
    model = joblib.load(MODEL_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    return model, schema["feature_columns"]


def prepare_features(
    df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Recreate the exact feature matrix used in training.

    Reuses train_model.build_features() for encoding and leakage-column removal
    (no duplicated preprocessing logic), then reindexes to feature_columns so
    column order is guaranteed to match what the trained model expects, even if
    get_dummies would otherwise produce a different order on a data subset.
    """
    X, y = build_features(df)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X, y


def compute_builtin_importance(
    model: RandomForestClassifier, feature_columns: list[str]
) -> list[dict]:
    """Rank features by the Random Forest's built-in impurity-based importance.

    Known limitation (surfaced by _check_signal_agreement below): impurity
    importance tends to overestimate continuous/high-cardinality features
    regardless of whether they're actually predictive. Shown for comparison,
    not used to drive explanations.
    """
    ranked = sorted(
        zip(feature_columns, model.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [{"feature": name, "importance": round(float(value), 6)} for name, value in ranked]


def compute_permutation_importance(
    model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series
) -> list[dict]:
    """Rank features by how much shuffling each one degrades ROC-AUC.

    This is the ranking explanations are built from - it measures actual
    predictive reliance rather than tree-construction bias. Evaluated on the
    full dataset with the final model: acceptable here because this produces a
    relative feature ranking, not a claimed generalization performance number
    (unlike Phase 2's confusion matrix, which used out-of-fold predictions for
    that reason).
    """
    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,
    )
    ranked = sorted(
        zip(X.columns, result.importances_mean),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [{"feature": name, "importance": round(float(value), 6)} for name, value in ranked]


def save_importance(builtin_ranking: list[dict], permutation_ranking: list[dict]) -> None:
    """Persist both rankings to models/feature_importance.json for reporting.

    Fails loudly (rather than writing a partial/empty file) if either ranking
    is empty, since a downstream reader (the report, a future script) silently
    getting malformed JSON is worse than an explicit error here.
    """
    if not builtin_ranking or not permutation_ranking:
        raise ValueError("Cannot save feature importance: one or both rankings are empty.")

    payload = {"built_in": builtin_ranking, "permutation": permutation_ranking}
    IMPORTANCE_PATH.write_text(json.dumps(payload, indent=2))


def dataset_statistics(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute mean/median/std for each raw numeric feature.

    Used only to phrase human-readable explanations (e.g. "above the dataset
    average"), never fed to the model.
    """
    stats = {}
    for column in NUMERIC_FEATURES:
        stats[column] = {
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
            "std": float(df[column].std()),
        }
    return stats


def _format_ranking_lines(ranking: list[dict]) -> list[str]:
    """Format a ranked importance list as '1. feature ............ 0.318' lines."""
    lines = []
    for position, item in enumerate(ranking, start=1):
        label = f"{position}. {item['feature']}"
        dots = "." * max(3, 32 - len(label))
        lines.append(f"{label} {dots} {item['importance']:.3f}")
    return lines


def print_global_summary(builtin_ranking: list[dict], permutation_ranking: list[dict]) -> None:
    """Print both global importance rankings, every feature, for comparison.

    Both are shown so a reader can see the two methods largely agree on the
    top drivers (queue_length, historical_avg_delay) - the disagreement is
    isolated to arrival_hour, surfaced separately in the sanity check.
    """
    print("=" * 49)
    print("Global Feature Importance")
    print("=" * 49)

    print("\nBuilt-in Importance\n")
    for line in _format_ranking_lines(builtin_ranking):
        print(line)

    print("\nPermutation Importance\n")
    for line in _format_ranking_lines(permutation_ranking):
        print(line)


def _meaningful_features(permutation_ranking: list[dict]) -> list[dict]:
    """Return the top permutation-ranked features that clear MIN_IMPORTANCE,
    capped at MAX_FEATURES_PER_EXPLANATION. This is the single source of truth
    for "which features are worth mentioning" - both the numeric and
    categorical branches of explain_shipment draw from it, so the filtering
    logic exists in exactly one place.
    """
    cleared = [item for item in permutation_ranking if item["importance"] >= MIN_IMPORTANCE]
    return cleared[:MAX_FEATURES_PER_EXPLANATION]


def _describe_numeric_feature(
    feature: str, raw_row: pd.Series, stats: dict[str, dict[str, float]]
) -> str | None:
    """Return a manager-facing sentence about a numeric feature if its value is
    significantly different from the dataset typical value, else None.

    Defensive: returns None (skip, don't crash) if the feature is missing from
    stats or the row, or if std is zero/unavailable.
    """
    feature_stats = stats.get(feature)
    if feature_stats is None or feature not in raw_row:
        return None

    mean = feature_stats.get("mean")
    std = feature_stats.get("std")
    if mean is None or not std:
        return None

    value = raw_row[feature]
    z_score = (value - mean) / std

    if z_score > SIGNIFICANCE_Z_THRESHOLD:
        return NUMERIC_PHRASES_HIGH.get(feature)
    if z_score < -SIGNIFICANCE_Z_THRESHOLD:
        return NUMERIC_PHRASES_LOW.get(feature)
    return None  # not significantly different from typical - not worth mentioning


def _describe_categorical_feature(feature: str, raw_row: pd.Series) -> str | None:
    """Return a manager-facing sentence for a one-hot categorical feature if it
    is this shipment's active category, else None.

    Defensive: returns None if the base column is missing from the row
    (handles a malformed/partial input row gracefully instead of raising).
    """
    base_column, separator, category = feature.partition("_")
    if not separator:
        return None
    if raw_row.get(base_column) != category:
        return None
    return CATEGORICAL_PHRASES.get((base_column, category))


def _risk_band(probability: float) -> str:
    """Classify a predicted probability into a manager-facing risk band.

    Three bands rather than a strict 0.5 cutoff, since a 0.51 and a 0.98 are
    operationally very different even though a binary classifier treats them
    the same.
    """
    if probability >= 0.7:
        return "High Delay Risk"
    if probability <= 0.3:
        return "Low Delay Risk"
    return "Moderate Delay Risk"


def explain_shipment(
    shipment_number: int,
    raw_row: pd.Series,
    probability: float,
    permutation_ranking: list[dict],
    stats: dict[str, dict[str, float]],
) -> str:
    """Build a plain-language explanation for one shipment, aimed at a logistics
    manager: only features the model demonstrably relies on (permutation
    importance >= MIN_IMPORTANCE) and whose value is notably different from
    typical for this shipment are mentioned. Never raises - falls back to a
    generic message if no feature clears the bar, so one odd shipment can't
    crash the whole report.
    """
    top_feature_name = permutation_ranking[0]["feature"] if permutation_ranking else None

    bullets: list[str] = []
    for item in _meaningful_features(permutation_ranking):
        feature = item["feature"]

        if feature in NUMERIC_FEATURES:
            sentence = _describe_numeric_feature(feature, raw_row, stats)
        else:
            sentence = _describe_categorical_feature(feature, raw_row)

        if sentence is None:
            continue  # value isn't notably different from typical - skip silently

        if feature == top_feature_name:
            sentence += " This is the single biggest factor in this prediction."

        bullets.append(sentence)

    if not bullets:
        bullets.append(
            "No single factor stands out significantly for this shipment; "
            "predicted risk reflects a combination of typical operating conditions."
        )

    risk_band = _risk_band(probability)
    closing = (
        f"Overall, this shipment is assessed as {risk_band.lower()} "
        f"({probability:.0%} predicted probability of delay)."
    )

    lines = [
        f"Shipment #{shipment_number}",
        "",
        "Prediction",
        "",
        f"{risk_band} ({probability:.0%})",
        "",
        "Explanation",
        "",
    ]
    lines.extend(f"* {bullet}" for bullet in bullets)
    lines.append("")
    lines.append(closing)
    return "\n".join(lines)


def _select_example_shipments(proba: np.ndarray) -> list[int]:
    """Pick two illustrative shipments: the highest and lowest predicted delay
    probability. Deterministic (no randomness) and gives a clear contrastive
    pair for the report/demo rather than two arbitrary rows.
    """
    return [int(np.argmax(proba)), int(np.argmin(proba))]


def _ranking_lookup(ranking: list[dict]) -> dict[str, tuple[int, float]]:
    """Build a {feature: (rank, importance)} lookup from a ranking list."""
    return {
        item["feature"]: (position, item["importance"])
        for position, item in enumerate(ranking, start=1)
    }


def _check_signal_agreement(builtin_ranking: list[dict], permutation_ranking: list[dict]) -> None:
    """Sanity check on SANITY_CHECK_FEATURE (arrival_hour): compare what the two
    importance methods say about a feature we know the ground truth for
    (dataset_design.md documents it as carrying no real signal).

    Three outcomes:
    - Built-in relatively high + permutation near zero: expected discrepancy,
      explained rather than just reported. Impurity-based importance in a
      Random Forest tends to overestimate continuous numerical features
      (more possible split points inflate their apparent importance) even
      when they have no real relationship with the target.
    - Both near zero: genuine agreement the feature carries little signal - OK.
    - Anything else (e.g. permutation itself finds signal): flagged as a
      stronger warning, since that would contradict the documented data
      design and deserves investigation, not a shrug.
    """
    builtin_lookup = _ranking_lookup(builtin_ranking)
    permutation_lookup = _ranking_lookup(permutation_ranking)

    if SANITY_CHECK_FEATURE not in builtin_lookup or SANITY_CHECK_FEATURE not in permutation_lookup:
        print(f"\nSanity check skipped: '{SANITY_CHECK_FEATURE}' not found in rankings.")
        return

    builtin_rank, builtin_value = builtin_lookup[SANITY_CHECK_FEATURE]
    permutation_rank, permutation_value = permutation_lookup[SANITY_CHECK_FEATURE]
    total = len(builtin_ranking)

    print("\n" + "=" * 49)
    print(f"Sanity Check: {SANITY_CHECK_FEATURE}")
    print("=" * 49)
    print(f"Built-in importance:     rank {builtin_rank}/{total}  (value={builtin_value:.3f})")
    print(
        f"Permutation importance:  rank {permutation_rank}/{total}  (value={permutation_value:.3f})"
    )
    print()

    permutation_is_near_zero = permutation_value < PERMUTATION_NEAR_ZERO
    builtin_is_relatively_high = builtin_value >= BUILTIN_RELATIVELY_HIGH

    if permutation_is_near_zero and builtin_is_relatively_high:
        print(
            f"Built-in feature importance ranks '{SANITY_CHECK_FEATURE}' higher than "
            "expected, while permutation importance shows almost no predictive "
            "contribution. This discrepancy is expected because impurity-based "
            "feature importance in Random Forest can overestimate continuous "
            "numerical features."
        )
    elif permutation_is_near_zero and not builtin_is_relatively_high:
        print(
            f"OK: both methods agree '{SANITY_CHECK_FEATURE}' carries little signal, as expected."
        )
    else:
        print(
            f"WARNING: permutation importance detects real predictive signal for "
            f"'{SANITY_CHECK_FEATURE}', which contradicts docs/dataset_design.md "
            "(it is documented as unused in the delay formula). Investigate for a "
            "data generation or training bug before trusting this model."
        )


def main() -> None:
    model, feature_columns = load_model()

    df = pd.read_csv(DATA_PATH)
    X, y = prepare_features(df, feature_columns)

    builtin_ranking = compute_builtin_importance(model, feature_columns)
    permutation_ranking = compute_permutation_importance(model, X, y)
    save_importance(builtin_ranking, permutation_ranking)

    print_global_summary(builtin_ranking, permutation_ranking)

    stats = dataset_statistics(df)
    proba = model.predict_proba(X)[:, 1]

    print("\n" + "=" * 49)
    print("Example Shipment Explanations")
    print("=" * 49 + "\n")

    for shipment_index in _select_example_shipments(proba):
        try:
            explanation = explain_shipment(
                shipment_number=shipment_index + 1,
                raw_row=df.iloc[shipment_index],
                probability=float(proba[shipment_index]),
                permutation_ranking=permutation_ranking,
                stats=stats,
            )
        except Exception as error:  # noqa: BLE001 - explanation generation must never crash
            explanation = (
                f"Shipment #{shipment_index + 1}\n\nUnable to generate explanation: {error}"
            )
        print(explanation)
        print()

    _check_signal_agreement(builtin_ranking, permutation_ranking)


if __name__ == "__main__":
    main()
