import duckdb


def build_analytics_dataset():

    conn = duckdb.connect(
        "data/hospital_readmissions.duckdb"
    )

    print("Creating analytics dataset...")

    conn.execute("""
    CREATE OR REPLACE TABLE ml_features AS

    WITH base AS (

        SELECT

            identificador,

            sexo,

            municipiosolicitante,
            municipioresidencia,
            municipioexecutante,

            executante,

            especialidade,
            tipoleito,
            carater,

            codigocid,
            motivoalta,

            admission_ts,
            discharge_ts,

            readmitted_30d_clean,
            eligible_for_readmission_model,
            
            DATEDIFF(
                'day',
                admission_ts,
                discharge_ts
            ) AS length_of_stay_days,

            ROW_NUMBER() OVER (
                PARTITION BY identificador
                ORDER BY admission_ts
            ) - 1 AS previous_hospitalizations,

            LAG(discharge_ts) OVER (
                PARTITION BY identificador
                ORDER BY admission_ts
            ) AS previous_discharge_ts,

            COALESCE(
                SUM(readmitted_30d_clean) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                    ROWS BETWEEN UNBOUNDED PRECEDING
                    AND 1 PRECEDING
                ),
                0
            ) AS previous_readmissions

        FROM hospitalization_target

    ),

    features AS (

        SELECT

            *,

            CASE

                WHEN previous_discharge_ts IS NULL
                THEN NULL

                ELSE DATEDIFF(
                    'day',
                    previous_discharge_ts,
                    admission_ts
                )

            END AS days_since_previous_hospitalization,

            CASE

                WHEN discharge_ts IS NULL
                THEN 1

                ELSE 0

            END AS missing_discharge,

            DATE_TRUNC(
                'month',
                admission_ts
            ) AS admission_month,

            YEAR(admission_ts)
                AS admission_year,

            QUARTER(admission_ts)
                AS admission_quarter,

            DAYOFWEEK(admission_ts)
                AS admission_weekday,

            CASE

                WHEN DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                ) <= 1
                    THEN '0-1 days'

                WHEN DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                ) <= 3
                    THEN '2-3 days'

                WHEN DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                ) <= 7
                    THEN '4-7 days'

                WHEN DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                ) <= 14
                    THEN '8-14 days'

                ELSE '15+ days'

            END AS los_bucket

        FROM base

    )

    SELECT

        *,

        COUNT(*) OVER (

            PARTITION BY identificador

            ORDER BY admission_ts

            RANGE BETWEEN
                INTERVAL 365 DAY PRECEDING
                AND CURRENT ROW

        ) AS admissions_last_365d

    FROM features
    """)

    print("\nDataset Summary:")

    summary = conn.execute("""
        SELECT

            COUNT(*) AS total_records,

            SUM(readmitted_30d_clean)
                AS readmissions,

            ROUND(
                AVG(readmitted_30d_clean) * 100,
                2
            ) AS readmission_rate,

            ROUND(
                AVG(length_of_stay_days),
                2
            ) AS avg_los

        FROM ml_features
    """).fetchdf()

    print(summary)

    print("\nFeature Schema:")

    schema = conn.execute("""
        DESCRIBE ml_features
    """).fetchdf()

    print(schema)

    conn.close()

    print("\nAnalytics dataset completed.")


if __name__ == "__main__":
    build_analytics_dataset()