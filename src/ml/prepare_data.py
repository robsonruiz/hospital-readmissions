from pathlib import Path

import duckdb
import pandas as pd


DB_PATH = Path("data/hospital_readmissions.duckdb")
TARGET = "unplanned_readmitted_30d"

# Columns that must not be used as predictors.
# admission_ts is retained temporarily only to perform the temporal split.
DROP_COLUMNS = [
    TARGET,
    "readmitted_30d_clean",
    "eligible_for_readmission_model",
    "readmitted_30d_raw",
    "next_admission_ts",
    "days_until_next_admission",
    "admission_ts",
    "discharge_ts",
    "previous_discharge_ts",
    "admission_month",
    "identificador",
]


def load_ml_data():
    """Load the modeling dataset from DuckDB."""
    con = duckdb.connect(DB_PATH, read_only=True)

    df = con.execute(
        """
        SELECT *
        FROM ml_features
        ORDER BY admission_ts
        """
    ).df()

    con.close()

    if df.empty:
        raise ValueError("ml_features is empty.")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found in ml_features."
        )

    if "admission_ts" not in df.columns:
        raise ValueError(
            "Column 'admission_ts' is required for temporal splitting."
        )

    df["admission_ts"] = pd.to_datetime(
        df["admission_ts"],
        errors="coerce"
    )

    missing_dates = df["admission_ts"].isna().sum()

    if missing_dates > 0:
        raise ValueError(
            f"{missing_dates} records have invalid/missing admission_ts. "
            "Temporal splitting cannot be performed safely."
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    if df[TARGET].isna().any():
        raise ValueError(
            f"Target '{TARGET}' contains missing values."
        )

    df[TARGET] = df[TARGET].astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            f"Target '{TARGET}' must contain only 0 and 1."
        )

    return df


def _get_temporal_cutoffs(df):
    """
    Determine temporal cutoffs close to 70% / 15% / 15%.

    The cutoffs are based on row positions after chronological sorting,
    but they are moved to the end of the corresponding calendar date.
    This prevents records from the same date from being split between
    different datasets.
    """
    dates = df["admission_ts"].dt.normalize()

    n = len(df)

    train_position = max(1, int(n * 0.70)) - 1
    validation_position = max(1, int(n * 0.85)) - 1

    train_cutoff_date = dates.iloc[train_position]
    validation_cutoff_date = dates.iloc[validation_position]

    # If both positions fall on the same date, move the validation
    # boundary forward to the next available date.
    if validation_cutoff_date <= train_cutoff_date:
        unique_dates = dates.drop_duplicates().sort_values().reset_index(drop=True)

        train_idx = unique_dates.searchsorted(train_cutoff_date)
        valid_idx = min(train_idx + 1, len(unique_dates) - 1)

        train_cutoff_date = unique_dates.iloc[train_idx]
        validation_cutoff_date = unique_dates.iloc[valid_idx]

    return train_cutoff_date, validation_cutoff_date


def prepare_data(df):
    """
    Perform a chronological train/validation/test split.

    Approximate proportions:
        Train      ~70%
        Validation ~15%
        Test       ~15%

    The split is temporal, not random.
    """
    df = df.sort_values("admission_ts").reset_index(drop=True).copy()

    train_cutoff, validation_cutoff = _get_temporal_cutoffs(df)

    dates = df["admission_ts"].dt.normalize()

    train_mask = dates <= train_cutoff
    validation_mask = (
        (dates > train_cutoff)
        & (dates <= validation_cutoff)
    )
    test_mask = dates > validation_cutoff

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError(
            "Temporal split produced an empty dataset. "
            "Check the date distribution in ml_features."
        )

    # Keep admission_ts available only for diagnostics.
    # It is removed from the actual model features below.
    y_train = train_df[TARGET].copy()
    y_valid = validation_df[TARGET].copy()
    y_test = test_df[TARGET].copy()

    X_train = train_df.drop(
        columns=[c for c in DROP_COLUMNS if c in train_df.columns]
    )
    X_valid = validation_df.drop(
        columns=[c for c in DROP_COLUMNS if c in validation_df.columns]
    )
    X_test = test_df.drop(
        columns=[c for c in DROP_COLUMNS if c in test_df.columns]
    )

    # Final safety check: the target must never remain in X.
    for X, name in [
        (X_train, "X_train"),
        (X_valid, "X_valid"),
        (X_test, "X_test"),
    ]:
        if TARGET in X.columns:
            raise ValueError(
                f"Target leakage detected: '{TARGET}' is present in {name}."
            )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        train_df,
        validation_df,
        test_df,
        train_cutoff,
        validation_cutoff,
    )


def print_split_summary(
    train_df,
    validation_df,
    test_df,
    train_cutoff,
    validation_cutoff,
):
    """Print the temporal split and target prevalence."""
    print("\n" + "=" * 70)
    print("TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    datasets = [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df),
    ]

    total = len(train_df) + len(validation_df) + len(test_df)

    for name, data in datasets:
        start = data["admission_ts"].min().date()
        end = data["admission_ts"].max().date()
        n = len(data)
        positives = int(data[TARGET].sum())
        prevalence = data[TARGET].mean() * 100

        print(f"\n{name}")
        print(f"  Period: {start} -> {end}")
        print(f"  Records: {n:,}")
        print(f"  Positive: {positives:,}")
        print(f"  Negative: {n - positives:,}")
        print(f"  Positive rate: {prevalence:.2f}%")
        print(f"  Share of total: {n / total * 100:.2f}%")

    print("\n" + "-" * 70)
    print(f"Train cutoff:      {train_cutoff.date()}")
    print(f"Validation cutoff: {validation_cutoff.date()}")
    print("-" * 70)

    # Explicit chronological safety checks.
    train_max = train_df["admission_ts"].max()
    valid_min = validation_df["admission_ts"].min()
    valid_max = validation_df["admission_ts"].max()
    test_min = test_df["admission_ts"].min()

    print("\nChronological checks:")
    print(f"  Train max < Validation min: {train_max < valid_min}")
    print(f"  Validation max < Test min:  {valid_max < test_min}")

    if not (train_max < valid_min and valid_max < test_min):
        raise ValueError(
            "Temporal ordering check failed."
        )

    print("=" * 70)


def main():
    print("1 - Loading ml_features...")
    df = load_ml_data()

    print(f"Loaded: {len(df):,} records")
    print(
        f"Date range: "
        f"{df['admission_ts'].min().date()} -> "
        f"{df['admission_ts'].max().date()}"
    )

    print("\n2 - Creating temporal split...")

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        train_df,
        validation_df,
        test_df,
        train_cutoff,
        validation_cutoff,
    ) = prepare_data(df)

    print_split_summary(
        train_df,
        validation_df,
        test_df,
        train_cutoff,
        validation_cutoff,
    )

    print("\nFeature shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_valid: {X_valid.shape}")
    print(f"  X_test:  {X_test.shape}")

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    )


if __name__ == "__main__":
    main()
