import duckdb


def build_readmission_target():

    conn = duckdb.connect(
        "data/hospital_readmissions.duckdb"
    )

    print("Creating clean episode table...")

    conn.execute("""
        CREATE OR REPLACE TABLE hospitalization_episodes_clean AS

        SELECT
            *,

            STRPTIME(
                datahorainternacao,
                '%d/%m/%Y %H:%M:%S'
            ) AS admission_ts,

            CASE
                WHEN datahoraalta IS NULL
                    OR datahoraalta = 'SEM DATAHORAALTA'
                THEN NULL

                ELSE STRPTIME(
                    datahoraalta,
                    '%d/%m/%Y %H:%M:%S'
                )
            END AS discharge_ts

        FROM hospitalization_episodes
    """)

    print("Creating readmission target...")

    conn.execute("""
        CREATE OR REPLACE TABLE hospitalization_target AS

        WITH ordered_admissions AS (

            SELECT
                *,

                LEAD(admission_ts) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                ) AS next_admission_ts,

                LEAD(executante) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                ) AS next_executante,

                LEAD(municipioexecutante) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                ) AS next_municipioexecutante

            FROM hospitalization_episodes_clean
        ),

        target_logic AS (

            SELECT
                *,

                DATEDIFF(
                    'day',
                    discharge_ts,
                    next_admission_ts
                ) AS days_until_next_admission,

                CASE
                    WHEN motivoalta = 'Transferência'
                    THEN 1
                    ELSE 0
                END AS is_transfer_discharge,

                CASE
                    WHEN motivoalta = 'Óbito'
                    THEN 1
                    ELSE 0
                END AS is_death_discharge

            FROM ordered_admissions
        )

        SELECT
            *,

            CASE
                WHEN next_admission_ts IS NULL
                    THEN 0

                WHEN discharge_ts IS NULL
                    THEN 0

                WHEN is_death_discharge = 1
                    THEN 0

                WHEN is_transfer_discharge = 1
                    THEN 0

                WHEN days_until_next_admission BETWEEN 0 AND 30
                    THEN 1

                ELSE 0

            END AS readmitted_30d_raw,

            CASE
                WHEN next_admission_ts IS NULL
                    THEN 0

                WHEN discharge_ts IS NULL
                    THEN 0

                WHEN is_death_discharge = 1
                    THEN 0

                WHEN is_transfer_discharge = 1
                    THEN 0

                WHEN days_until_next_admission BETWEEN 2 AND 30
                    THEN 1

                ELSE 0

            END AS readmitted_30d_clean

        FROM target_logic
    """)

    summary = conn.execute("""
        SELECT
            readmitted_30d_clean,
            COUNT(*) AS records
        FROM hospitalization_target
        GROUP BY readmitted_30d_clean
        ORDER BY readmitted_30d_clean
    """).fetchdf()

    print("\nReadmission target distribution:")
    print(summary)

    comparison = conn.execute("""
        SELECT
            SUM(readmitted_30d_raw) AS raw_readmissions,
            SUM(readmitted_30d_clean) AS clean_readmissions
        FROM hospitalization_target
    """).fetchdf()

    print("\nRaw vs Clean target:")
    print(comparison)

    excluded = conn.execute("""
        SELECT
            motivoalta,
            COUNT(*) AS episodes
        FROM hospitalization_target
        WHERE motivoalta IN (
            'Transferência',
            'Óbito'
        )
        GROUP BY motivoalta
        ORDER BY episodes DESC
    """).fetchdf()

    print("\nExcluded discharge reasons:")
    print(excluded)

    total = conn.execute("""
        SELECT COUNT(*)
        FROM hospitalization_target
    """).fetchone()[0]

    positives = conn.execute("""
        SELECT COUNT(*)
        FROM hospitalization_target
        WHERE readmitted_30d_clean = 1
    """).fetchone()[0]

    print(f"\nTotal episodes: {total:,}")
    print(f"Readmissions within 30 days clean: {positives:,}")
    print(f"Rate: {(positives / total) * 100:.2f}%")

    conn.close()

    print("\nTarget creation completed.")


if __name__ == "__main__":
    build_readmission_target()