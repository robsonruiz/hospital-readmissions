from pathlib import Path
import sys

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DB_PATH = PROJECT_ROOT / "data" / "hospital_readmissions.duckdb"
DATASET_PATH = (
    PROJECT_ROOT
    / "models"
    / "trajectory_experiment"
    / "dataset_A_baseline.parquet"
)

TARGET = "unplanned_readmitted_30d"

EXPECTED_FEATURES = [
    "sexo",
    "executante",
    "especialidade",
    "tipoleito",
    "carater",
    "codigocid",
    "motivoalta",
    "length_of_stay_days",
    "previous_hospitalizations",
    "previous_unplanned_readmissions",
    "admissions_last_30d",
    "admissions_last_90d",
    "admissions_last_365d",
    "days_since_previous_hospitalization",
    "los_bucket",
]


def load_official():
    con = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )

    df = con.execute(
        """
        SELECT *
        FROM ml_features
        ORDER BY admission_ts
        """
    ).df()

    con.close()

    return df


def main():
    print()
    print("=" * 80)
    print("BASELINE EQUIVALENCE AUDIT")
    print("=" * 80)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset A not found:\n{DATASET_PATH}"
        )

    print()
    print("1. Loading official ml_features...", flush=True)

    official = load_official()

    print(
        f"Official rows: {len(official):,}",
        flush=True,
    )

    print()
    print("2. Loading dataset A...", flush=True)

    dataset_a = pd.read_parquet(
        DATASET_PATH
    )

    print(
        f"Dataset A rows: {len(dataset_a):,}",
        flush=True,
    )

    print()
    print("3. Comparing columns...", flush=True)

    official_features = [
        column
        for column in EXPECTED_FEATURES
        if column in official.columns
    ]

    a_features = [
        column
        for column in EXPECTED_FEATURES
        if column in dataset_a.columns
    ]

    print()
    print("Expected baseline features:")

    for column in EXPECTED_FEATURES:
        print(
            f"  {'OK' if column in dataset_a.columns else 'MISSING'} "
            f"{column}"
        )

    extra_a = sorted(
        set(dataset_a.columns)
        - set(EXPECTED_FEATURES)
        - {
            "identificador",
            "admission_ts",
            "discharge_ts",
            TARGET,
        }
    )

    if extra_a:
        print()
        print("Extra feature columns in Dataset A:")

        for column in extra_a:
            print(
                f"  - {column}"
            )

    print()
    print("4. Comparing row counts...")

    if len(official) == len(dataset_a):
        print("  OK: same number of rows")
    else:
        print(
            "  DIFFERENCE: "
            f"official={len(official):,}, "
            f"A={len(dataset_a):,}"
        )

    print()
    print("5. Comparing patient/episode keys...")

    key_columns = [
        "identificador",
        "admission_ts",
        "discharge_ts",
    ]

    for column in key_columns:
        if column not in official.columns:
            print(
                f"  Official missing: {column}"
            )
            continue

        if column not in dataset_a.columns:
            print(
                f"  Dataset A missing: {column}"
            )
            continue

        left = official[column].astype(str)
        right = dataset_a[column].astype(str)

        same = (
            len(left) == len(right)
            and left.reset_index(drop=True).equals(
                right.reset_index(drop=True)
            )
        )

        print(
            f"  {column}: "
            f"{'OK' if same else 'DIFFERENT'}"
        )

    print()
    print("6. Comparing target...")

    official_target = (
        pd.to_numeric(
            official[TARGET],
            errors="coerce",
        )
        .reset_index(drop=True)
    )

    a_target = (
        pd.to_numeric(
            dataset_a[TARGET],
            errors="coerce",
        )
        .reset_index(drop=True)
    )

    same_target = (
        len(official_target) == len(a_target)
        and official_target.equals(a_target)
    )

    print(
        f"  Target values: "
        f"{'OK' if same_target else 'DIFFERENT'}"
    )

    print(
        f"  Official positive rate: "
        f"{official_target.mean():.4%}"
    )

    print(
        f"  Dataset A positive rate: "
        f"{a_target.mean():.4%}"
    )

    print()
    print("7. Comparing baseline feature values...")

    comparison_rows = []

    for column in EXPECTED_FEATURES:
        if (
            column not in official.columns
            or column not in dataset_a.columns
        ):
            comparison_rows.append(
                {
                    "feature": column,
                    "status": "MISSING",
                    "differences": None,
                    "max_numeric_difference": None,
                }
            )
            continue

        left = official[column].reset_index(
            drop=True
        )

        right = dataset_a[column].reset_index(
            drop=True
        )

        if (
            pd.api.types.is_numeric_dtype(left)
            and pd.api.types.is_numeric_dtype(right)
        ):
            left_num = pd.to_numeric(
                left,
                errors="coerce",
            )

            right_num = pd.to_numeric(
                right,
                errors="coerce",
            )

            both_nan = (
                left_num.isna()
                & right_num.isna()
            )

            differences = (
                (
                    left_num - right_num
                ).abs() > 1e-10
            ) & ~both_nan

            max_difference = (
                (
                    left_num - right_num
                ).abs()
                .where(~both_nan)
                .max()
            )

        else:
            left_str = left.astype(str)
            right_str = right.astype(str)

            differences = (
                left_str != right_str
            )

            max_difference = None

        n_differences = int(
            differences.sum()
        )

        comparison_rows.append(
            {
                "feature": column,
                "status": (
                    "OK"
                    if n_differences == 0
                    else "DIFFERENT"
                ),
                "differences": n_differences,
                "max_numeric_difference": max_difference,
            }
        )

        print(
            f"  {column}: "
            f"{'OK' if n_differences == 0 else 'DIFFERENT'} "
            f"({n_differences:,} differences)"
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    output_dir = (
        PROJECT_ROOT
        / "models"
        / "trajectory_experiment"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "baseline_equivalence_audit.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    all_features_equal = (
        (comparison["status"] == "OK").all()
    )

    all_rows_equal = (
        len(official) == len(dataset_a)
    )

    all_target_equal = same_target

    if (
        all_features_equal
        and all_rows_equal
        and all_target_equal
    ):
        print(
            "BASELINE A IS EQUIVALENT TO THE OFFICIAL ML DATASET."
        )
        print()
        print(
            "The trajectory experiment is methodologically "
            "comparable to the original baseline."
        )
    else:
        print(
            "BASELINE A IS NOT FULLY EQUIVALENT "
            "TO THE OFFICIAL ML DATASET."
        )
        print()
        print(
            "Do NOT proceed to TEST yet."
        )
        print(
            "The differences above must be investigated first."
        )

    print()
    print(
        f"Audit saved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()