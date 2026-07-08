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
                ) AS next_admission_ts

            FROM hospitalization_episodes_clean

        )

        SELECT

            *,

            DATEDIFF(
                'day',
                discharge_ts,
                next_admission_ts
            ) AS days_until_next_admission,

        CASE

            WHEN motivoalta = 'Óbito'
                THEN 0

            WHEN next_admission_ts IS NULL
                THEN 0

            WHEN discharge_ts IS NULL
                THEN 0

            WHEN DATEDIFF(
                'day',
                discharge_ts,
                next_admission_ts
            ) BETWEEN 2 AND 30
                THEN 1

            ELSE 0

        END AS readmitted_30d_clean

        FROM ordered_admissions
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

    conn.close()

    print("\nTarget creation completed.")


if __name__ == "__main__":
    build_readmission_target()