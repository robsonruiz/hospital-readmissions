from pathlib import Path

import duckdb


DB_PATH = Path("data/hospital_readmissions.duckdb")
OUTPUT_DIR = Path("data/quality_audit/readmission_planning_details")

TARGET_TABLE = "hospitalization_target"
CLEAN_TABLE = "hospitalization_episodes_clean"


def run_audit(
    connection: duckdb.DuckDBPyConnection,
    title: str,
    filename: str,
    query: str,
) -> None:
    result = connection.execute(query).fetchdf()

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(result.to_string(index=False))

    result.to_csv(
        OUTPUT_DIR / filename,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Banco DuckDB não encontrado: {DB_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )

    try:
        run_audit(
            connection,
            "RESUMO DO TARGET NÃO PLANEJADO",
            "01_unplanned_target_summary.csv",
            f"""
            SELECT
                COUNT(*) AS total_episodes,

                COUNT(*) FILTER (
                    WHERE eligible_for_readmission_model = 1
                ) AS eligible_episodes,

                COUNT(*) FILTER (
                    WHERE eligible_for_readmission_model = 0
                ) AS ineligible_episodes,

                COUNT(*) FILTER (
                    WHERE unplanned_readmitted_30d = 1
                ) AS unplanned_readmissions,

                COUNT(*) FILTER (
                    WHERE unplanned_readmitted_30d = 0
                ) AS episodes_without_unplanned_readmission,

                ROUND(
                    COUNT(*) FILTER (
                        WHERE unplanned_readmitted_30d = 1
                    ) * 100.0
                    / NULLIF(
                        COUNT(*) FILTER (
                            WHERE eligible_for_readmission_model = 1
                        ),
                        0
                    ),
                    2
                ) AS unplanned_readmission_rate

            FROM {TARGET_TABLE}
            """,
        )

        run_audit(
            connection,
            "REINTERNAÇÕES GERAIS VERSUS NÃO PLANEJADAS",
            "02_general_vs_unplanned.csv",
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE eligible_for_readmission_model = 1
                      AND days_until_next_admission BETWEEN 2 AND 30
                ) AS general_readmissions_2_30d,

                COUNT(*) FILTER (
                    WHERE unplanned_readmitted_30d = 1
                ) AS unplanned_readmissions_2_30d,

                COUNT(*) FILTER (
                    WHERE eligible_for_readmission_model = 1
                      AND days_until_next_admission BETWEEN 2 AND 30
                      AND unplanned_readmitted_30d = 0
                ) AS readmissions_not_classified_as_unplanned,

                ROUND(
                    COUNT(*) FILTER (
                        WHERE eligible_for_readmission_model = 1
                          AND days_until_next_admission BETWEEN 2 AND 30
                          AND unplanned_readmitted_30d = 0
                    ) * 100.0
                    / NULLIF(
                        COUNT(*) FILTER (
                            WHERE eligible_for_readmission_model = 1
                              AND days_until_next_admission BETWEEN 2 AND 30
                        ),
                        0
                    ),
                    2
                ) AS excluded_from_general_readmissions_percentage

            FROM {TARGET_TABLE}
            """,
        )

        run_audit(
            connection,
            "VALIDAÇÃO DE CONSISTÊNCIA",
            "03_target_validation.csv",
            f"""
            SELECT
                (
                    SELECT COUNT(*)
                    FROM {CLEAN_TABLE}
                ) AS clean_episodes,

                (
                    SELECT COUNT(*)
                    FROM {TARGET_TABLE}
                ) AS target_episodes,

                COUNT(*) FILTER (
                    WHERE eligible_for_readmission_model = 0
                      AND unplanned_readmitted_30d IS NOT NULL
                ) AS ineligible_with_non_null_target,

                COUNT(*) FILTER (
                    WHERE eligible_for_readmission_model = 1
                      AND unplanned_readmitted_30d IS NULL
                ) AS eligible_with_null_target,

                COUNT(*) FILTER (
                    WHERE unplanned_readmitted_30d = 1
                      AND days_until_next_admission NOT BETWEEN 2 AND 30
                ) AS positives_outside_window,

                COUNT(*) - COUNT(
                    DISTINCT (
                        identificador,
                        admission_ts
                    )
                ) AS duplicate_episodes

            FROM {TARGET_TABLE}
            """,
        )

        run_audit(
            connection,
            "MOTIVOS DE NÃO ELEGIBILIDADE",
            "04_non_eligibility_reasons.csv",
            f"""
            SELECT
                CASE
                    WHEN motivoalta = 'Óbito'
                    THEN 'Óbito'

                    WHEN motivoalta = 'Transferência'
                    THEN 'Transferência'

                    WHEN discharge_ts IS NULL
                    THEN 'Alta ausente'

                    ELSE 'Outro'
                END AS reason,

                COUNT(*) AS episodes

            FROM {TARGET_TABLE}

            WHERE eligible_for_readmission_model = 0

            GROUP BY 1
            ORDER BY episodes DESC
            """,
        )

        run_audit(
            connection,
            "DISTRIBUIÇÃO DO INTERVALO DAS REINTERNAÇÕES NÃO PLANEJADAS",
            "05_unplanned_interval_distribution.csv",
            f"""
            SELECT
                days_until_next_admission,
                COUNT(*) AS unplanned_readmissions

            FROM {TARGET_TABLE}

            WHERE unplanned_readmitted_30d = 1

            GROUP BY days_until_next_admission
            ORDER BY days_until_next_admission
            """,
        )

    finally:
        connection.close()

    print(
        "\nAuditoria concluída. Resultados em: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()