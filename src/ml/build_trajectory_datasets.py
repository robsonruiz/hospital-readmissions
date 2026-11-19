from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "hospital_readmissions.duckdb"
TRAJECTORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "trajectory_audit"
    / "06_combined_pre_admission_trajectory.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "models" / "trajectory_experiment"

TARGET = "unplanned_readmitted_30d"

BASELINE_FEATURES = [
    "previous_hospitalizations",
    "previous_unplanned_readmissions",
    "admissions_last_30d",
    "admissions_last_90d",
    "admissions_last_365d",
    "days_since_previous_hospitalization",
]

APS_FEATURES = [
    "consultas_aps_last_30d",
    "consultas_aps_last_90d",
    "consultas_aps_last_365d",
    "exames_regulados_aps_last_30d",
    "exames_regulados_aps_last_90d",
    "exames_regulados_aps_last_365d",
    "exames_nao_regulados_aps_last_30d",
    "exames_nao_regulados_aps_last_90d",
    "exames_nao_regulados_aps_last_365d",
    "solicitacoes_aps_last_30d",
    "solicitacoes_aps_last_90d",
    "solicitacoes_aps_last_365d",
]

TRAJECTORY_KEYS = [
    "episode_id",
    "identificador",
    "admission_ts",
]

EXCLUDED_TRAJECTORY_COLUMNS = {
    "episode_id",
    "identificador",
    "admission_ts",
}


def load_official_ml_features():
    print()
    print("=" * 80)
    print("1. LOADING OFFICIAL ML FEATURES")
    print("=" * 80)

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

    required_columns = [
        "identificador",
        "admission_ts",
        TARGET,
        *BASELINE_FEATURES,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns in ml_features: "
            + ", ".join(missing)
        )

    df["identificador"] = (
        df["identificador"]
        .astype("string")
        .str.strip()
    )

    df["admission_ts"] = pd.to_datetime(
        df["admission_ts"],
        errors="coerce",
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    ).astype("Int64")

    if df["admission_ts"].isna().any():
        raise ValueError(
            "ml_features contains invalid admission_ts values."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            f"Target '{TARGET}' contains missing values."
        )

    print(f"Official ml_features rows: {len(df):,}")
    print(
        f"Official ml_features patients: "
        f"{df['identificador'].nunique():,}"
    )

    return df


def load_trajectory_data():
    print()
    print("=" * 80)
    print("2. LOADING TRAJECTORY DATA")
    print("=" * 80)

    if not TRAJECTORY_PATH.exists():
        raise FileNotFoundError(
            f"Trajectory file not found:\n{TRAJECTORY_PATH}"
        )

    df = pd.read_csv(
        TRAJECTORY_PATH,
        low_memory=False,
    )

    if df.empty:
        raise ValueError("Trajectory dataset is empty.")

    required_columns = [
        "episode_id",
        "identificador",
        "admission_ts",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required trajectory columns: "
            + ", ".join(missing)
        )

    df["identificador"] = (
        df["identificador"]
        .astype("string")
        .str.strip()
    )

    df["admission_ts"] = pd.to_datetime(
        df["admission_ts"],
        errors="coerce",
    )

    if df["admission_ts"].isna().any():
        raise ValueError(
            "Trajectory dataset contains invalid admission_ts values."
        )

    print(f"Trajectory rows: {len(df):,}")
    print(
        f"Trajectory patients: "
        f"{df['identificador'].nunique():,}"
    )

    return df


def identify_trajectory_features(trajectory):
    print()
    print("=" * 80)
    print("3. TRAJECTORY FEATURE CANDIDATES")
    print("=" * 80)

    candidate_columns = [
        column
        for column in trajectory.columns
        if column not in EXCLUDED_TRAJECTORY_COLUMNS
    ]

    print(
        f"Candidate columns: "
        f"{len(candidate_columns)}"
    )

    for column in candidate_columns:
        print(f"  - {column}")

    missing_aps = [
        column
        for column in APS_FEATURES
        if column not in trajectory.columns
    ]

    if missing_aps:
        raise ValueError(
            "Expected APS columns are missing from trajectory data:\n"
            + "\n".join(f"  - {column}" for column in missing_aps)
        )

    return candidate_columns


def validate_keys(baseline, trajectory):
    print()
    print("=" * 80)
    print("4. VALIDATING TRAJECTORY KEYS")
    print("=" * 80)

    baseline_duplicates = baseline.duplicated(
        ["identificador", "admission_ts"]
    ).sum()

    trajectory_duplicates = trajectory.duplicated(
        ["identificador", "admission_ts"]
    ).sum()

    print(
        f"Baseline duplicate keys: "
        f"{baseline_duplicates:,}"
    )

    print(
        f"Trajectory duplicate keys: "
        f"{trajectory_duplicates:,}"
    )

    if baseline_duplicates > 0:
        raise ValueError(
            "Baseline contains duplicate "
            "identificador + admission_ts keys."
        )

    if trajectory_duplicates > 0:
        raise ValueError(
            "Trajectory contains duplicate "
            "identificador + admission_ts keys."
        )


def merge_datasets(baseline, trajectory):
    print()
    print("=" * 80)
    print("5. MERGING")
    print("=" * 80)

    print(f"Baseline rows: {len(baseline):,}")
    print(f"Trajectory rows: {len(trajectory):,}")

    merged = baseline.merge(
        trajectory,
        on=[
            "identificador",
            "admission_ts",
        ],
        how="left",
        validate="one_to_one",
        suffixes=("", "_trajectory"),
    )

    print(f"Merged rows: {len(merged):,}")

    if len(merged) != len(baseline):
        raise ValueError(
            "Merge changed the number of baseline rows."
        )

    missing_trajectory = (
        merged["episode_id"]
        .isna()
        .sum()
    )

    print(
        f"Admissions without trajectory match: "
        f"{missing_trajectory:,}"
    )

    if missing_trajectory > 0:
        raise ValueError(
            "Some ml_features episodes could not be matched "
            "to the trajectory dataset."
        )

    return merged


def build_dataset(
    merged,
    feature_columns,
    dataset_name,
):
    print()
    print("=" * 80)
    print(f"BUILDING {dataset_name}")
    print("=" * 80)

    required_columns = [
        "identificador",
        "admission_ts",
        TARGET,
        *feature_columns,
    ]

    missing = [
        column
        for column in required_columns
        if column not in merged.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns for {dataset_name}: "
            + ", ".join(missing)
        )

    dataset = merged[
        required_columns
    ].copy()

    dataset = dataset.sort_values(
        "admission_ts"
    ).reset_index(drop=True)

    # Keep identifiers only for validation and traceability.
    # They are not model features.
    model_features = [
        column
        for column in feature_columns
        if column not in {
            "identificador",
            "admission_ts",
            "episode_id",
        }
    ]

    dataset = dataset[
        [
            "identificador",
            "admission_ts",
            *model_features,
            TARGET,
        ]
    ]

    print(f"Rows: {len(dataset):,}")
    print(
        f"Features: {len(model_features)}"
    )

    for feature in model_features:
        print(f"  - {feature}")

    output_path = (
        OUTPUT_DIR
        / f"{dataset_name.lower()}.parquet"
    )

    dataset.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")

    return dataset


def validate_baseline(
    official_ml_features,
    dataset_a,
):
    print()
    print("=" * 80)
    print("7. BASELINE VALIDATION")
    print("=" * 80)

    official_keys = official_ml_features[
        [
            "identificador",
            "admission_ts",
            TARGET,
        ]
    ].copy()

    dataset_a_keys = dataset_a[
        [
            "identificador",
            "admission_ts",
            TARGET,
        ]
    ].copy()

    print(
        f"Official ml_features rows: "
        f"{len(official_keys):,}"
    )

    print(
        f"Dataset A rows: "
        f"{len(dataset_a_keys):,}"
    )

    official_duplicates = official_keys.duplicated(
        [
            "identificador",
            "admission_ts",
        ]
    ).sum()

    dataset_a_duplicates = dataset_a_keys.duplicated(
        [
            "identificador",
            "admission_ts",
        ]
    ).sum()

    print(
        f"Official duplicate keys: "
        f"{official_duplicates:,}"
    )

    print(
        f"Dataset A duplicate keys: "
        f"{dataset_a_duplicates:,}"
    )

    if official_duplicates > 0:
        raise ValueError(
            "Official ml_features contains duplicate keys."
        )

    if dataset_a_duplicates > 0:
        raise ValueError(
            "Dataset A contains duplicate keys."
        )

    comparison = official_keys.merge(
        dataset_a_keys,
        on=[
            "identificador",
            "admission_ts",
        ],
        how="outer",
        suffixes=(
            "_official",
            "_dataset_a",
        ),
        indicator=True,
    )

    missing_in_a = (
        comparison["_merge"]
        == "left_only"
    ).sum()

    extra_in_a = (
        comparison["_merge"]
        == "right_only"
    ).sum()

    print(
        f"Missing episodes in Dataset A: "
        f"{missing_in_a:,}"
    )

    print(
        f"Extra episodes in Dataset A: "
        f"{extra_in_a:,}"
    )

    if missing_in_a > 0 or extra_in_a > 0:
        raise ValueError(
            "Dataset A does not contain exactly the same "
            "hospitalization episodes as ml_features."
        )

    official_target = comparison[
        f"{TARGET}_official"
    ]

    dataset_a_target = comparison[
        f"{TARGET}_dataset_a"
    ]

    target_diff = (
        official_target.ne(dataset_a_target)
        & ~(
            official_target.isna()
            & dataset_a_target.isna()
        )
    )

    target_differences = target_diff.sum()

    print(
        f"Target differences: "
        f"{target_differences:,}"
    )

    if target_differences > 0:
        print()
        print("First target differences:")

        print(
            comparison.loc[
                target_diff,
                [
                    "identificador",
                    "admission_ts",
                    f"{TARGET}_official",
                    f"{TARGET}_dataset_a",
                ],
            ]
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "Dataset A target differs from ml_features."
        )

    print(
        "✓ Dataset A target matches "
        "ml_features by hospitalization episode."
    )


def validate_dataset_structure(
    dataset,
    expected_features,
    dataset_name,
):
    actual_features = [
        column
        for column in dataset.columns
        if column not in {
            "identificador",
            "admission_ts",
            TARGET,
        }
    ]

    if actual_features != expected_features:
        raise ValueError(
            f"{dataset_name} feature structure is incorrect.\n"
            f"Expected: {expected_features}\n"
            f"Actual: {actual_features}"
        )

    print()
    print(
        f"✓ {dataset_name} structure validated: "
        f"{len(actual_features)} features."
    )


def main():
    print()
    print("=" * 80)
    print("BUILDING TRAJECTORY EXPERIMENT DATASETS")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 1. Official baseline
    # ------------------------------------------------------------------
    official_ml_features = (
        load_official_ml_features()
    )

    # ------------------------------------------------------------------
    # 2. Trajectory data
    # ------------------------------------------------------------------
    trajectory = load_trajectory_data()

    # ------------------------------------------------------------------
    # 3. Identify trajectory features
    # ------------------------------------------------------------------
    trajectory_features = (
        identify_trajectory_features(
            trajectory
        )
    )

    # ------------------------------------------------------------------
    # 4. Validate keys
    # ------------------------------------------------------------------
    validate_keys(
        official_ml_features,
        trajectory,
    )

    # ------------------------------------------------------------------
    # 5. Merge
    # ------------------------------------------------------------------
    merged = merge_datasets(
        official_ml_features,
        trajectory,
    )

    # ------------------------------------------------------------------
    # 6. Build datasets
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("6. BUILDING DATASETS")
    print("=" * 80)

    # Dataset A:
    # Baseline hospital history only.
    dataset_a = build_dataset(
        merged=merged,
        feature_columns=BASELINE_FEATURES,
        dataset_name="A_baseline",
    )

    # Dataset C:
    # Baseline + explicitly classified APS utilization.
    dataset_c_features = (
        BASELINE_FEATURES
        + APS_FEATURES
    )

    dataset_c = build_dataset(
        merged=merged,
        feature_columns=dataset_c_features,
        dataset_name="C_aps",
    )

    # Dataset B:
    # Baseline + complete trajectory.
    #
    # episode_id is deliberately excluded because it is
    # only a technical identifier created by the audit.
    dataset_b_features = (
        BASELINE_FEATURES
        + [
            column
            for column in trajectory_features
            if column != "episode_id"
        ]
    )

    dataset_b = build_dataset(
        merged=merged,
        feature_columns=dataset_b_features,
        dataset_name="B_trajectory",
    )

    # ------------------------------------------------------------------
    # 7. Validate Dataset A against official ml_features
    # ------------------------------------------------------------------
    validate_baseline(
        official_ml_features,
        dataset_a,
    )

    # ------------------------------------------------------------------
    # 8. Validate dataset structures
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("8. DATASET STRUCTURE VALIDATION")
    print("=" * 80)

    validate_dataset_structure(
        dataset_a,
        BASELINE_FEATURES,
        "Dataset A",
    )

    validate_dataset_structure(
        dataset_c,
        dataset_c_features,
        "Dataset C",
    )

    validate_dataset_structure(
        dataset_b,
        dataset_b_features,
        "Dataset B",
    )

    # ------------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("9. FINAL SUMMARY")
    print("=" * 80)

    print()
    print(
        f"Dataset A - Baseline: "
        f"{len(BASELINE_FEATURES)} features"
    )

    print(
        f"Dataset C - Baseline + APS: "
        f"{len(dataset_c_features)} features"
    )

    print(
        f"Dataset B - Baseline + Trajectory: "
        f"{len(dataset_b_features)} features"
    )

    print()
    print("Files generated:")

    print(
        f"  {OUTPUT_DIR / 'a_baseline.parquet'}"
    )

    print(
        f"  {OUTPUT_DIR / 'c_aps.parquet'}"
    )

    print(
        f"  {OUTPUT_DIR / 'b_trajectory.parquet'}"
    )

    print()
    print("✓ Dataset construction completed successfully.")
    print("✓ No model was trained.")
    print("✓ No validation/test performance was evaluated.")
    print("=" * 80)


if __name__ == "__main__":
    main()