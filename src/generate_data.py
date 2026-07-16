from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path("src/data/port_data.csv")
RANDOM_SEED = 42
NUM_ROWS = 500
DELAY_THRESHOLD = 24

# Scales the operational effect terms (queue, utilization, interaction) down from
# their raw formula weights. Without this, mean simulated delay (~30h) sits above
# DELAY_THRESHOLD, producing a ~70% positive rate. Calibrated by simulation so the
# positive class lands at ~37%, inside the 30-45% minority-class target documented
# in docs/planning.md #2. The 24h threshold is left untouched since it's the
# operationally meaningful boundary (see planning.md); only how easily the formula
# crosses it changes.
OPERATIONAL_EFFECT_SCALE = 0.55

VESSEL_TYPES = [
    "Container",
    "Bulk",
    "Tanker",
]

VESSEL_DELAY = {
    "Container": 2,
    "Bulk": 4,
    "Tanker": 6,
}

WEATHER_TYPES = [
    "Clear",
    "Rain",
    "Storm",
]

WEATHER_DELAY = {
    "Clear": 0,
    "Rain": 4,
    "Storm": 10,
}

WEATHER_PROBABILITIES = [
    0.70,
    0.20,
    0.10,
]

rng = np.random.default_rng(RANDOM_SEED)


def generate_dtset(rng: np.random.Generator) -> pd.DataFrame:
    vessel_type = rng.choice(
        VESSEL_TYPES,
        size=NUM_ROWS,
        p=[0.5, 0.3, 0.2],
    )

    weather = rng.choice(
        WEATHER_TYPES,
        size=NUM_ROWS,
        p=WEATHER_PROBABILITIES,
    )

    queue_length = rng.integers(
        low=0,
        high=26,
        size=NUM_ROWS,
    )

    port_utilization = rng.uniform(
        low=60,
        high=100,
        size=NUM_ROWS,
    ).round(1)

    historical_avg_delay = rng.uniform(
        low=0,
        high=12,
        size=NUM_ROWS,
    ).round(2)

    arrival_hour = rng.integers(
        low=0,
        high=24,
        size=NUM_ROWS,
    )

    return pd.DataFrame(
        {
            "vessel_type": vessel_type,
            "weather": weather,
            "queue_length": queue_length,
            "port_utilization": port_utilization,
            "historical_avg_delay": historical_avg_delay,
            "arrival_hour": arrival_hour,
        }
    )


def calculate_delay(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    weather_delay = df["weather"].map(WEATHER_DELAY)

    vessel_delay = df["vessel_type"].map(VESSEL_DELAY)

    queue_effect = df["queue_length"] * 0.8 * OPERATIONAL_EFFECT_SCALE

    utilization_effect = (df["port_utilization"] - 60) * 0.25 * OPERATIONAL_EFFECT_SCALE

    interaction_effect = (
        df["queue_length"] * (df["port_utilization"] / 100) * 0.4 * OPERATIONAL_EFFECT_SCALE
    )

    noise = rng.normal(
        loc=0,
        scale=2,
        size=len(df),
    )

    delay_hours = (
        df["historical_avg_delay"]
        + weather_delay
        + vessel_delay
        + queue_effect
        + utilization_effect
        + interaction_effect
        + noise
    ).clip(lower=0)

    df["delay_hours"] = delay_hours.round(2)

    df["is_delayed"] = (df["delay_hours"] > DELAY_THRESHOLD).astype(int)

    return df


def save_dataset(df: pd.DataFrame):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )


def print_summary(df: pd.DataFrame):
    print("=" * 60)
    print("Synthetic Dataset Summary")
    print("=" * 60)

    print(f"Rows: {len(df)}")
    print(f"Delayed Rate: {df['is_delayed'].mean() * 100:.2f}%")

    print("\nWeather Distribution")
    print(df["weather"].value_counts(normalize=True))

    print("\nVessel Distribution")
    print(df["vessel_type"].value_counts(normalize=True))

    print("\nFirst Five Rows")
    print(df.head())


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df = generate_dtset(rng)

    df = calculate_delay(df, rng)

    save_dataset(df)

    print_summary(df)


if __name__ == "__main__":
    main()
