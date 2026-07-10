from pathlib import Path

import duckdb
import pandas as pd


DATABASE_PATH = "data/hospital_readmissions.duckdb"
OUTPUT_PATH = Path("reports/audit/overlapping_episodes.csv")


def audit_overlapping_episodes() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(DATABASE_PATH)

    query = """
        WITH ordered_episodes AS (
            SELECT
                identificador,
                admission_ts,
                discharge_ts,
                executante,
                municipioexecutante,
                especialidade,
                tipoleito,
                carater,
                codigocid,
                motivoalta,

                LEAD(admission_ts) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                ) AS next_admission_ts,

                LEAD(discharge_ts) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                ) AS next_discharge_ts,

                LEAD(executante) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                ) AS next_executante,

                LEAD(municipioexecutante) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                ) AS next_municipioexecutante,

                LEAD(especialidade) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                ) AS next_especialidade,

                LEAD(tipoleito) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                ) AS next_tipoleito,

                LEAD(codigocid) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                ) AS next_codigocid,

                LEAD(motivoalta) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                ) AS next_motivoalta

            FROM hospitalization_target
        )

        SELECT
            identificador,

            admission_ts,
            discharge_ts,

            next_admission_ts,
            next_discharge_ts,

            DATEDIFF(
                'day',
                next_admission_ts,
                discharge_ts
            ) AS overlap_days,

            DATEDIFF(
                'hour',
                next_admission_ts,
                discharge_ts
            ) AS overlap_hours,

            executante,
            next_executante,

            municipioexecutante,
            next_municipioexecutante,

            especialidade,
            next_especialidade,

            tipoleito,
            next_tipoleito,

            codigocid,
            next_codigocid,

            motivoalta,
            next_motivoalta,

            CASE
                WHEN executante = next_executante
                THEN 1
                ELSE 0
            END AS same_hospital,

            CASE
                WHEN especialidade = next_especialidade
                THEN 1
                ELSE 0
            END AS same_specialty,

            CASE
                WHEN motivoalta = 'Transferência'
                THEN 1
                ELSE 0
            END AS transfer_discharge

        FROM ordered_episodes

        WHERE
            discharge_ts IS NOT NULL
            AND next_admission_ts IS NOT NULL
            AND next_admission_ts < discharge_ts

        ORDER BY
            overlap_hours DESC,
            identificador,
            admission_ts
    """

    overlaps = conn.execute(query).fetchdf()

    print("\n=== Overlapping Episode Summary ===")

    print(
        conn.execute("""
            WITH ordered_episodes AS (
                SELECT
                    identificador,
                    admission_ts,
                    discharge_ts,

                    LEAD(admission_ts) OVER (
                        PARTITION BY identificador
                        ORDER BY admission_ts
                    ) AS next_admission_ts,

                    executante,

                    LEAD(executante) OVER (
                        PARTITION BY identificador
                        ORDER BY admission_ts
                    ) AS next_executante,

                    especialidade,

                    LEAD(especialidade) OVER (
                        PARTITION BY identificador
                        ORDER BY admission_ts
                    ) AS next_especialidade,

                    motivoalta

                FROM hospitalization_target
            )

            SELECT
                COUNT(*) AS overlapping_episodes,

                COUNT(DISTINCT identificador)
                    AS affected_patients,

                SUM(
                    CASE
                        WHEN executante = next_executante
                        THEN 1
                        ELSE 0
                    END
                ) AS same_hospital,

                SUM(
                    CASE
                        WHEN executante <> next_executante
                        THEN 1
                        ELSE 0
                    END
                ) AS different_hospital,

                SUM(
                    CASE
                        WHEN especialidade = next_especialidade
                        THEN 1
                        ELSE 0
                    END
                ) AS same_specialty,

                SUM(
                    CASE
                        WHEN motivoalta = 'Transferência'
                        THEN 1
                        ELSE 0
                    END
                ) AS transfer_discharge

            FROM ordered_episodes

            WHERE
                discharge_ts IS NOT NULL
                AND next_admission_ts IS NOT NULL
                AND next_admission_ts < discharge_ts
        """).fetchdf()
    )

    print("\n=== Overlap Duration Distribution ===")

    print(
        conn.execute("""
            WITH ordered_episodes AS (
                SELECT
                    identificador,
                    admission_ts,
                    discharge_ts,

                    LEAD(admission_ts) OVER (
                        PARTITION BY identificador
                        ORDER BY admission_ts
                    ) AS next_admission_ts

                FROM hospitalization_target
            )

            SELECT
                CASE
                    WHEN DATEDIFF(
                        'hour',
                        next_admission_ts,
                        discharge_ts
                    ) <= 6
                    THEN '0-6 hours'

                    WHEN DATEDIFF(
                        'hour',
                        next_admission_ts,
                        discharge_ts
                    ) <= 24
                    THEN '7-24 hours'

                    WHEN DATEDIFF(
                        'day',
                        next_admission_ts,
                        discharge_ts
                    ) <= 3
                    THEN '2-3 days'

                    WHEN DATEDIFF(
                        'day',
                        next_admission_ts,
                        discharge_ts
                    ) <= 7
                    THEN '4-7 days'

                    ELSE '8+ days'
                END AS overlap_bucket,

                COUNT(*) AS episodes

            FROM ordered_episodes

            WHERE
                discharge_ts IS NOT NULL
                AND next_admission_ts IS NOT NULL
                AND next_admission_ts < discharge_ts

            GROUP BY overlap_bucket

            ORDER BY episodes DESC
        """).fetchdf()
    )

    print("\n=== Sample Overlapping Episodes ===")

    columns_to_display = [
        "identificador",
        "admission_ts",
        "discharge_ts",
        "next_admission_ts",
        "next_discharge_ts",
        "overlap_days",
        "overlap_hours",
        "executante",
        "next_executante",
        "especialidade",
        "next_especialidade",
        "motivoalta",
        "same_hospital",
        "same_specialty",
        "transfer_discharge",
    ]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 220)

    print(overlaps[columns_to_display].head(50))

    overlaps.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    conn.close()

    print(f"\nFull audit saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    audit_overlapping_episodes()