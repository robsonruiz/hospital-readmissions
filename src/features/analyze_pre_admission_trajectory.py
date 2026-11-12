from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "hospital_readmissions.duckdb"

TRAJECTORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "trajectory_audit"
    / "06_combined_pre_admission_trajectory.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "trajectory_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

ID_COLUMN = "episode_id"
TARGET_COLUMN = "unplanned_readmitted_30d"

TRAJECTORY_FEATURES = [
    "consultas_last_30d",
    "consultas_last_90d",
    "consultas_last_365d",
    "consultas_aps_last_30d",
    "consultas_aps_last_90d",
    "consultas_aps_last_365d",
    "days_since_last_consultas",
    "exames_regulados_last_30d",
    "exames_regulados_last_90d",
    "exames_regulados_last_365d",
    "exames_regulados_aps_last_30d",
    "exames_regulados_aps_last_90d",
    "exames_regulados_aps_last_365d",
    "days_since_last_exames_regulados",
    "exames_nao_regulados_last_30d",
    "exames_nao_regulados_last_90d",
    "exames_nao_regulados_last_365d",
    "exames_nao_regulados_aps_last_30d",
    "exames_nao_regulados_aps_last_90d",
    "exames_nao_regulados_aps_last_365d",
    "days_since_last_exames_nao_regulados",
    "solicitacoes_last_30d",
    "solicitacoes_last_90d",
    "solicitacoes_last_365d",
    "solicitacoes_aps_last_30d",
    "solicitacoes_aps_last_90d",
    "solicitacoes_aps_last_365d",
    "days_since_last_solicitacoes",
]

# =============================================================================
# HELPERS
# =============================================================================

def normalize_identifier(series: pd.Series) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
            }
        )
    )


def load_targets() -> pd.DataFrame:
    print()
    print("=" * 80)
    print("1. LOADING HOSPITALIZATION TARGET")
    print("=" * 80)

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )

    df = con.execute(
        """
        SELECT
            identificador,
            admission_ts,
            discharge_ts,
            eligible_for_readmission_model,
            unplanned_readmitted_30d
        FROM hospitalization_target
        WHERE eligible_for_readmission_model = 1
          AND unplanned_readmitted_30d IS NOT NULL
          AND identificador IS NOT NULL
          AND admission_ts IS NOT NULL
        ORDER BY admission_ts
        """
    ).df()

    con.close()

    df["identificador"] = normalize_identifier(
        df["identificador"]
    )

    df["admission_ts"] = pd.to_datetime(
        df["admission_ts"],
        errors="coerce",
    ).astype("datetime64[ns]")

    df = df.dropna(
        subset=[
            "identificador",
            "admission_ts",
        ]
    )

    print(
        f"Episodes: {len(df):,}"
    )

    print(
        f"Patients: "
        f"{df['identificador'].nunique():,}"
    )

    print()
    print("Target distribution:")

    print(
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print()

    print(
        "Target rate: "
        f"{df[TARGET_COLUMN].mean():.4%}"
    )

    return df

def load_trajectory() -> pd.DataFrame:
    print()
    print("=" * 80)
    print("2. LOADING PRE-ADMISSION TRAJECTORY")
    print("=" * 80)

    if not TRAJECTORY_PATH.exists():
        raise FileNotFoundError(
            f"Trajectory file not found:\n{TRAJECTORY_PATH}"
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

    print()

    return df

def prepare_analysis_dataset(
    targets: pd.DataFrame,
    trajectory: pd.DataFrame,
) -> pd.DataFrame:
    print()
    print("=" * 80)
    print("3. BUILDING ANALYSIS DATASET")
    print("=" * 80)

    missing_features = [
        column
        for column in TRAJECTORY_FEATURES
        if column not in trajectory.columns
    ]

    if missing_features:
        print()
        print(
            "WARNING: The following trajectory features "
            "were not found:"
        )

        for column in missing_features:
            print(f"  - {column}")

        print()

    available_features = [
        column
        for column in TRAJECTORY_FEATURES
        if column in trajectory.columns
    ]

    trajectory_small = trajectory[
        [
            ID_COLUMN,
            "identificador",
            "admission_ts",
            *available_features,
        ]
    ].copy()

    targets_small = targets[
        [
            "identificador",
            "admission_ts",
            TARGET_COLUMN,
        ]
    ].copy()

    trajectory_small["identificador"] = normalize_identifier(
        trajectory_small["identificador"]
    )

    targets_small["identificador"] = normalize_identifier(
        targets_small["identificador"]
    )

    trajectory_small["admission_ts"] = pd.to_datetime(
        trajectory_small["admission_ts"],
        errors="coerce",
    ).astype("datetime64[ns]")

    targets_small["admission_ts"] = pd.to_datetime(
        targets_small["admission_ts"],
        errors="coerce",
    ).astype("datetime64[ns]")

    trajectory_small = trajectory_small.dropna(
        subset=[
            "identificador",
            "admission_ts",
        ]
    )

    targets_small = targets_small.dropna(
        subset=[
            "identificador",
            "admission_ts",
        ]
    )

    # Check whether the trajectory contains more than one
    # record for the same hospitalization.
    duplicate_keys = trajectory_small.duplicated(
        subset=[
            "identificador",
            "admission_ts",
        ],
        keep=False,
    )

    duplicate_count = duplicate_keys.sum()

    print(
        f"Duplicate trajectory keys: "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:
        print(
            "WARNING: duplicated "
            "(identificador, admission_ts) keys found."
        )

        trajectory_small = (
            trajectory_small
            .sort_values(
                [
                    "identificador",
                    "admission_ts",
                    ID_COLUMN,
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

    # Merge target information into the trajectory
    # using the actual hospitalization identity.
    df = trajectory_small.merge(
        targets_small,
        on=[
            "identificador",
            "admission_ts",
        ],
        how="inner",
        validate="one_to_one",
    )

    for column in available_features:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print(
        f"Trajectory episodes: "
        f"{len(trajectory_small):,}"
    )

    print(
        f"Eligible target episodes: "
        f"{len(targets_small):,}"
    )

    print(
        f"Matched episodes: "
        f"{len(df):,}"
    )

    print(
        f"Unmatched trajectory episodes: "
        f"{len(trajectory_small) - len(df):,}"
    )

    print(
        f"Unmatched target episodes: "
        f"{len(targets_small) - len(df):,}"
    )

    if len(df) > 0:
        print()
        print("Matched target distribution:")

        print(
            df[TARGET_COLUMN]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

    return df

def calculate_iqr(
    series: pd.Series,
) -> float:
    return (
        series.quantile(0.75)
        - series.quantile(0.25)
    )


def calculate_rank_biserial(
    group_0: pd.Series,
    group_1: pd.Series,
    u_statistic: float,
) -> float:
    n0 = len(group_0)
    n1 = len(group_1)

    if n0 == 0 or n1 == 0:
        return np.nan

    return (
        (2 * u_statistic)
        / (n0 * n1)
    ) - 1


def safe_auc(
    y: pd.Series,
    x: pd.Series,
) -> float:
    valid = (
        y.notna()
        & x.notna()
    )

    y_valid = y.loc[valid]
    x_valid = x.loc[valid]

    if y_valid.nunique() < 2:
        return np.nan

    if x_valid.nunique() < 2:
        return 0.5

    try:
        return roc_auc_score(
            y_valid,
            x_valid,
        )
    except ValueError:
        return np.nan


# =============================================================================
# STATISTICAL COMPARISON
# =============================================================================

def compare_groups(
    df: pd.DataFrame,
) -> pd.DataFrame:
    print()
    print("=" * 80)
    print("4. COMPARING TARGET 0 VS TARGET 1")
    print("=" * 80)

    results = []

    target_0 = df[
        df[TARGET_COLUMN] == 0
    ]

    target_1 = df[
        df[TARGET_COLUMN] == 1
    ]

    print(
        f"Target 0: {len(target_0):,}"
    )

    print(
        f"Target 1: {len(target_1):,}"
    )

    for feature in TRAJECTORY_FEATURES:
        if feature not in df.columns:
            continue

        x0 = target_0[feature].dropna()
        x1 = target_1[feature].dropna()

        n0 = len(x0)
        n1 = len(x1)

        if n0 == 0 or n1 == 0:
            continue

        median_0 = x0.median()
        median_1 = x1.median()

        mean_0 = x0.mean()
        mean_1 = x1.mean()

        iqr_0 = calculate_iqr(x0)
        iqr_1 = calculate_iqr(x1)

        positive_rate_0 = (
            (x0 > 0).mean()
        )

        positive_rate_1 = (
            (x1 > 0).mean()
        )

        try:
            u_statistic, p_value = mannwhitneyu(
                x0,
                x1,
                alternative="two-sided",
            )
        except ValueError:
            u_statistic = np.nan
            p_value = np.nan

        effect_size = calculate_rank_biserial(
            x0,
            x1,
            u_statistic,
        )

        auc = safe_auc(
            df[TARGET_COLUMN],
            df[feature],
        )

        results.append(
            {
                "feature": feature,
                "n_target_0": n0,
                "n_target_1": n1,
                "mean_target_0": mean_0,
                "mean_target_1": mean_1,
                "median_target_0": median_0,
                "median_target_1": median_1,
                "iqr_target_0": iqr_0,
                "iqr_target_1": iqr_1,
                "positive_rate_target_0": positive_rate_0,
                "positive_rate_target_1": positive_rate_1,
                "u_statistic": u_statistic,
                "p_value": p_value,
                "rank_biserial_effect": effect_size,
                "individual_roc_auc": auc,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # Multiple-testing correction.
    valid_p = results_df["p_value"].notna()

    results_df["p_value_fdr"] = np.nan

    if valid_p.any():
        adjusted = multipletests(
            results_df.loc[
                valid_p,
                "p_value",
            ],
            method="fdr_bh",
        )[1]

        results_df.loc[
            valid_p,
            "p_value_fdr",
        ] = adjusted

    results_df["absolute_effect"] = (
        results_df["rank_biserial_effect"]
        .abs()
    )

    results_df["auc_distance_from_random"] = (
        results_df["individual_roc_auc"]
        .sub(0.5)
        .abs()
    )

    results_df = results_df.sort_values(
        [
            "absolute_effect",
            "auc_distance_from_random",
        ],
        ascending=False,
    ).reset_index(drop=True)

    return results_df


# =============================================================================
# BINARY UTILIZATION COMPARISON
# =============================================================================

def compare_any_event(
    df: pd.DataFrame,
) -> pd.DataFrame:
    print()
    print("=" * 80)
    print("5. COMPARING ANY-EVENT UTILIZATION")
    print("=" * 80)

    results = []

    target_0 = df[
        df[TARGET_COLUMN] == 0
    ]

    target_1 = df[
        df[TARGET_COLUMN] == 1
    ]

    for feature in TRAJECTORY_FEATURES:
        if feature not in df.columns:
            continue

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        binary = (
            values > 0
        ).astype("float")

        binary = binary.where(
            values.notna()
        )

        group_0 = binary.loc[
            target_0.index
        ].dropna()

        group_1 = binary.loc[
            target_1.index
        ].dropna()

        if len(group_0) == 0 or len(group_1) == 0:
            continue

        rate_0 = group_0.mean()
        rate_1 = group_1.mean()

        absolute_difference = (
            rate_1 - rate_0
        )

        relative_risk = np.nan

        if rate_0 > 0:
            relative_risk = (
                rate_1 / rate_0
            )

        try:
            table = pd.crosstab(
                binary,
                df[TARGET_COLUMN],
            )

            if (
                0 in table.index
                and 1 in table.index
                and 0 in table.columns
                and 1 in table.columns
            ):
                a = table.loc[1, 1]
                b = table.loc[1, 0]
                c = table.loc[0, 1]
                d = table.loc[0, 0]

                odds_ratio = (
                    (a * d)
                    / (b * c)
                    if b > 0 and c > 0
                    else np.nan
                )
            else:
                odds_ratio = np.nan

        except Exception:
            odds_ratio = np.nan

        results.append(
            {
                "feature": feature,
                "any_event_target_0": rate_0,
                "any_event_target_1": rate_1,
                "absolute_difference": absolute_difference,
                "relative_risk": relative_risk,
                "odds_ratio": odds_ratio,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df["absolute_difference_pct_points"] = (
        results_df["absolute_difference"]
        * 100
    )

    results_df = results_df.sort_values(
        "absolute_difference",
        ascending=False,
    ).reset_index(drop=True)

    return results_df


# =============================================================================
# SAVE FEATURE RANKING
# =============================================================================

def create_feature_ranking(
    statistical_results: pd.DataFrame,
    binary_results: pd.DataFrame,
) -> pd.DataFrame:
    ranking = statistical_results[
        [
            "feature",
            "median_target_0",
            "median_target_1",
            "iqr_target_0",
            "iqr_target_1",
            "positive_rate_target_0",
            "positive_rate_target_1",
            "rank_biserial_effect",
            "absolute_effect",
            "p_value",
            "p_value_fdr",
            "individual_roc_auc",
            "auc_distance_from_random",
        ]
    ].copy()

    binary_small = binary_results[
        [
            "feature",
            "absolute_difference",
            "absolute_difference_pct_points",
            "relative_risk",
            "odds_ratio",
        ]
    ]

    ranking = ranking.merge(
        binary_small,
        on="feature",
        how="left",
    )

    ranking["fdr_significant"] = (
        ranking["p_value_fdr"] < 0.05
    )

    ranking["effect_strength"] = pd.cut(
        ranking["absolute_effect"],
        bins=[
            -np.inf,
            0.10,
            0.20,
            0.30,
            np.inf,
        ],
        labels=[
            "negligible",
            "small",
            "moderate",
            "large",
        ],
    )

    ranking = ranking.sort_values(
        [
            "absolute_effect",
            "auc_distance_from_random",
        ],
        ascending=False,
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        range(
            1,
            len(ranking) + 1,
        ),
    )

    return ranking


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_top_results(
    ranking: pd.DataFrame,
) -> None:
    print()
    print("=" * 80)
    print("6. TOP TRAJECTORY FEATURES")
    print("=" * 80)

    columns = [
        "rank",
        "feature",
        "median_target_0",
        "median_target_1",
        "positive_rate_target_0",
        "positive_rate_target_1",
        "rank_biserial_effect",
        "p_value_fdr",
        "individual_roc_auc",
    ]

    display = ranking[
        columns
    ].copy()

    print(
        display.head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print()
    print("=" * 80)
    print("PRE-ADMISSION TRAJECTORY: TARGET 0 VS TARGET 1")
    print("=" * 80)

    targets = load_targets()

    trajectory = load_trajectory()

    df = prepare_analysis_dataset(
        targets,
        trajectory,
    )

    statistical_results = compare_groups(
        df
    )

    binary_results = compare_any_event(
        df
    )

    ranking = create_feature_ranking(
        statistical_results,
        binary_results,
    )

    # Save analysis dataset.
    df.to_csv(
        OUTPUT_DIR
        / "01_trajectory_target_analysis_dataset.csv",
        index=False,
    )

    # Save statistical comparison.
    statistical_results.to_csv(
        OUTPUT_DIR
        / "02_trajectory_statistical_comparison.csv",
        index=False,
    )

    # Save binary utilization comparison.
    binary_results.to_csv(
        OUTPUT_DIR
        / "03_trajectory_any_event_comparison.csv",
        index=False,
    )

    # Save final feature ranking.
    ranking.to_csv(
        OUTPUT_DIR
        / "04_trajectory_feature_ranking.csv",
        index=False,
    )

    print_top_results(
        ranking
    )

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        OUTPUT_DIR
        / "01_trajectory_target_analysis_dataset.csv"
    )

    print(
        OUTPUT_DIR
        / "02_trajectory_statistical_comparison.csv"
    )

    print(
        OUTPUT_DIR
        / "03_trajectory_any_event_comparison.csv"
    )

    print(
        OUTPUT_DIR
        / "04_trajectory_feature_ranking.csv"
    )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()