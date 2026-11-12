from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
TRAJECTORY_DIR = MODEL_DIR / "trajectory_audit"
OUTPUT_DIR = MODEL_DIR / "trajectory_experiment"

ML_FEATURES_PATH = DATA_DIR / "ml_features.parquet"
TRAJECTORY_PATH = (
    TRAJECTORY_DIR
    / "06_combined_pre_admission_trajectory.csv"
)


TARGET = "unplanned_readmitted_30d"

BASELINE_FEATURES = [
    "previous_hospitalizations",
    "previous_unplanned_readmissions",
    "admissions_last_30d",
    "admissions_last_90d",
    "admissions_last_365d",
    "days_since_previous_hospitalization",
]

TRAJECTORY_GENERAL_FEATURES = [
    "consultas_last_30d",
    "consultas_last_90d",
    "consultas_last_365d",
    "consultas_aps_last_30d",
    "consultas_aps_last_90d",
    "consultas_aps_last_365d",
    "solicitacoes_last_30d",
    "solicitacoes_last_90d",
    "solicitacoes_last_365d",
]

TRAJECTORY_APS_FEATURES = [
    "consultas_aps_last_30d",
    "consultas_aps_last_90d",
    "consultas_aps_last_365d",
]

def load_ml_features() -> pd.DataFrame:
    print()
    print("=" * 80)
    print("1. LOADING CURRENT ML FEATURES")
    print("=" * 80)

    db_path = (
        PROJECT_ROOT
        / "data"
        / "hospital_readmissions.duckdb"
    )

    if not db_path.exists():
        raise FileNotFoundError(
            f"DuckDB database not found:\n"
            f"{db_path}"
        )

    import duckdb

    con = duckdb.connect(
        str(db_path),
        read_only=True,
    )

    tables = con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = 'ml_features'
        """
    ).fetchall()

    if not tables:
        con.close()

        raise ValueError(
            "Table 'ml_features' was not found "
            "in the DuckDB database."
        )

    df = con.execute(
        """
        SELECT *
        FROM ml_features
        """
    ).df()

    con.close()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    print(
        "Features available:"
    )

    for column in df.columns:
        print(
            f"  - {column}"
        )

    return df

def load_trajectory() -> pd.DataFrame:
    print()
    print("=" * 80)
    print("2. LOADING PRE-ADMISSION TRAJECTORY")
    print("=" * 80)

    if not TRAJECTORY_PATH.exists():
        raise FileNotFoundError(
            f"Trajectory file not found:\n"
            f"{TRAJECTORY_PATH}"
        )

    df = pd.read_csv(
        TRAJECTORY_PATH,
        low_memory=False,
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    return df


def normalize_keys(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df["identificador"] = (
        df["identificador"]
        .astype("string")
        .str.strip()
    )

    df["admission_ts"] = pd.to_datetime(
        df["admission_ts"],
        errors="coerce",
    ).astype("datetime64[ns]")

    return df


def merge_trajectory(
    ml_features: pd.DataFrame,
    trajectory: pd.DataFrame,
) -> pd.DataFrame:
    print()
    print("=" * 80)
    print("3. MERGING TRAJECTORY WITH ML FEATURES")
    print("=" * 80)

    ml_features = normalize_keys(
        ml_features
    )

    trajectory = normalize_keys(
        trajectory
    )

    required_ml = [
        "identificador",
        "admission_ts",
        TARGET,
        *BASELINE_FEATURES,
    ]

    missing_ml = [
        column
        for column in required_ml
        if column not in ml_features.columns
    ]

    if missing_ml:
        raise ValueError(
            "Missing columns in ml_features:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_ml
            )
        )

    trajectory_features = [
        column
        for column in (
            TRAJECTORY_GENERAL_FEATURES
        )
        if column in trajectory.columns
    ]

    missing_trajectory = [
        column
        for column in (
            TRAJECTORY_GENERAL_FEATURES
        )
        if column not in trajectory.columns
    ]

    if missing_trajectory:
        print()
        print(
            "WARNING: trajectory features "
            "not found:"
        )

        for column in missing_trajectory:
            print(
                f"  - {column}"
            )

    trajectory_small = trajectory[
        [
            "episode_id",
            "identificador",
            "admission_ts",
            *trajectory_features,
        ]
    ].copy()

    trajectory_small = (
        trajectory_small
        .dropna(
            subset=[
                "identificador",
                "admission_ts",
            ]
        )
        .drop_duplicates(
            subset=[
                "identificador",
                "admission_ts",
            ],
            keep="first",
        )
    )

    ml_features = (
        ml_features
        .dropna(
            subset=[
                "identificador",
                "admission_ts",
            ]
        )
    )

    print(
        f"ML feature rows: "
        f"{len(ml_features):,}"
    )

    print(
        f"Trajectory rows: "
        f"{len(trajectory_small):,}"
    )

    merged = ml_features.merge(
        trajectory_small[
            [
                "identificador",
                "admission_ts",
                *trajectory_features,
            ]
        ],
        on=[
            "identificador",
            "admission_ts",
        ],
        how="left",
        validate="one_to_one",
    )

    print(
        f"Merged rows: "
        f"{len(merged):,}"
    )

    for column in trajectory_features:
        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        ).fillna(0)

    return merged


def build_dataset(
    df: pd.DataFrame,
    features: list[str],
    name: str,
) -> None:
    print()
    print(
        f"Building dataset: {name}"
    )

    missing = [
        column
        for column in features
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features for {name}:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    columns = [
        column
        for column in df.columns
        if column not in {
            TARGET,
            "identificador",
            "admission_ts",
        }
    ]

    # Keep the original ML feature set for baseline,
    # plus only the selected trajectory variables.
    selected = [
        column
        for column in columns
        if column in features
    ]

    output = df[
        [
            "identificador",
            "admission_ts",
            *selected,
            TARGET,
        ]
    ].copy()

    output = output.sort_values(
        "admission_ts"
    ).reset_index(
        drop=True
    )

    output_path = (
        OUTPUT_DIR
        / f"{name}.parquet"
    )

    output.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"Features: {len(selected)}"
    )

    print(
        f"Rows: {len(output):,}"
    )

    print(
        f"Saved: {output_path}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ml_features = load_ml_features()

    trajectory = load_trajectory()

    merged = merge_trajectory(
        ml_features,
        trajectory,
    )

    # A — Current baseline.
    build_dataset(
        merged,
        BASELINE_FEATURES,
        "dataset_A_baseline",
    )

    # B — Hospital history + general trajectory.
    build_dataset(
        merged,
        (
            BASELINE_FEATURES
            + TRAJECTORY_GENERAL_FEATURES
        ),
        "dataset_B_trajectory",
    )

    # C — Hospital history + primary-care consultations.
    build_dataset(
        merged,
        (
            BASELINE_FEATURES
            + TRAJECTORY_APS_FEATURES
        ),
        "dataset_C_aps",
    )

    print()
    print("=" * 80)
    print("DATASET BUILD COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()