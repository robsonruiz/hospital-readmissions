import duckdb


DATABASE_PATH = "data/hospital_readmissions.duckdb"


def build_analytics_dataset() -> None:
    conn = duckdb.connect(DATABASE_PATH)

    try:
        print("Creating analytics dataset...")

        conn.execute("""
            CREATE OR REPLACE TABLE ml_features AS

            WITH history AS (
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
                    eligible_for_readmission_model,
                    unplanned_readmitted_30d,

                    DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) AS length_of_stay_days,

                    ROW_NUMBER() OVER (
                        PARTITION BY identificador
                        ORDER BY admission_ts
                    ) - 1 AS previous_hospitalizations,

                    COALESCE(
                        SUM(
                            COALESCE(
                                unplanned_readmitted_30d,
                                0
                            )
                        ) OVER (
                            PARTITION BY identificador
                            ORDER BY admission_ts
                            ROWS BETWEEN UNBOUNDED PRECEDING
                                AND 1 PRECEDING
                        ),
                        0
                    ) AS previous_unplanned_readmissions,

                    COUNT(*) OVER (
                        PARTITION BY identificador
                        ORDER BY admission_ts
                        RANGE BETWEEN INTERVAL 30 DAY PRECEDING
                            AND INTERVAL 1 MICROSECOND PRECEDING
                    ) AS admissions_last_30d,

                    COUNT(*) OVER (
                        PARTITION BY identificador
                        ORDER BY admission_ts
                        RANGE BETWEEN INTERVAL 90 DAY PRECEDING
                            AND INTERVAL 1 MICROSECOND PRECEDING
                    ) AS admissions_last_90d,

                    COUNT(*) OVER (
                        PARTITION BY identificador
                        ORDER BY admission_ts
                        RANGE BETWEEN INTERVAL 365 DAY PRECEDING
                            AND INTERVAL 1 MICROSECOND PRECEDING
                    ) AS admissions_last_365d

                FROM hospitalization_target
            )

            SELECT
                * EXCLUDE (
                    eligible_for_readmission_model,
                    unplanned_readmitted_30d
                ),

                CASE
                    WHEN previous_hospitalizations = 0
                    THEN 0

                    ELSE COALESCE(
                        DATEDIFF(
                            'day',
                            (
                                SELECT MAX(h2.discharge_ts)
                                FROM hospitalization_target h2
                                WHERE h2.identificador = history.identificador
                                  AND h2.discharge_ts IS NOT NULL
                                  AND h2.discharge_ts < history.admission_ts
                            ),
                            admission_ts
                        ),
                        0
                    )
                END AS days_since_previous_hospitalization,

                DATE_TRUNC(
                    'month',
                    admission_ts
                ) AS admission_month,

                YEAR(admission_ts) AS admission_year,

                QUARTER(admission_ts) AS admission_quarter,

                DAYOFWEEK(admission_ts) AS admission_weekday,

                CASE
                    WHEN length_of_stay_days <= 1
                    THEN '0-1 days'

                    WHEN length_of_stay_days <= 3
                    THEN '2-3 days'

                    WHEN length_of_stay_days <= 7
                    THEN '4-7 days'

                    WHEN length_of_stay_days <= 14
                    THEN '8-14 days'

                    ELSE '15+ days'
                END AS los_bucket,

                CAST(
                    unplanned_readmitted_30d
                    AS INTEGER
                ) AS unplanned_readmitted_30d

            FROM history

            WHERE eligible_for_readmission_model = 1
        """)

        eligible_count = conn.execute("""
            SELECT COUNT(*)
            FROM hospitalization_target
            WHERE eligible_for_readmission_model = 1
        """).fetchone()[0]

        analytics_count = conn.execute("""
            SELECT COUNT(*)
            FROM ml_features
        """).fetchone()[0]

        if eligible_count != analytics_count:
            raise ValueError(
                "Eligible episode count changed: "
                f"{eligible_count:,} -> {analytics_count:,}"
            )

        validation = conn.execute("""
            SELECT
                MIN(days_since_previous_hospitalization)
                    AS min_days,

                MAX(days_since_previous_hospitalization)
                    AS max_days,

                SUM(
                    CASE
                        WHEN days_since_previous_hospitalization < 0
                        THEN 1
                        ELSE 0
                    END
                ) AS negative_days,

                SUM(
                    CASE
                        WHEN previous_hospitalizations = 0
                             AND days_since_previous_hospitalization <> 0
                        THEN 1
                        ELSE 0
                    END
                ) AS inconsistent_first_episodes

            FROM ml_features
        """).fetchdf()

        print("\nDays-since-previous-hospitalization validation:")
        print(validation)

        summary = conn.execute("""
            SELECT
                COUNT(*) AS total_records,

                SUM(unplanned_readmitted_30d)
                    AS unplanned_readmissions,

                ROUND(
                    AVG(unplanned_readmitted_30d) * 100,
                    2
                ) AS unplanned_readmission_rate,

                ROUND(
                    AVG(length_of_stay_days),
                    2
                ) AS avg_length_of_stay_days

            FROM ml_features
        """).fetchdf()

        print("\nDataset summary:")
        print(summary)

        print("\nFeature schema:")
        print(
            conn.execute(
                "DESCRIBE ml_features"
            ).fetchdf()
        )

        print("\nAnalytics dataset completed.")

    finally:
        conn.close()


if __name__ == "__main__":
    build_analytics_dataset()