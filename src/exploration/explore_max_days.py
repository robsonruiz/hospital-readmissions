import duckdb


DATABASE_PATH = "data/hospital_readmissions.duckdb"


conn = duckdb.connect(DATABASE_PATH)

try:
    conn.execute("""
        WITH history AS (
            SELECT
                identificador,
                admission_ts,
                discharge_ts,

                LAG(discharge_ts) OVER (
                    PARTITION BY identificador
                    ORDER BY admission_ts
                ) AS previous_discharge_ts

            FROM hospitalization_target

            WHERE eligible_for_readmission_model = 1
        )

        SELECT
            identificador,
            admission_ts,
            previous_discharge_ts,
            DATEDIFF(
                'day',
                previous_discharge_ts,
                admission_ts
            ) AS days_since_previous_hospitalization

        FROM history

        WHERE previous_discharge_ts IS NOT NULL
          AND DATEDIFF(
                'day',
                previous_discharge_ts,
                admission_ts
              ) < 0

        ORDER BY days_since_previous_hospitalization

        LIMIT 30
    """).fetchdf().to_csv(
        "negative_days_examples.csv",
        index=False
    )

    print(
        conn.execute("""
            WITH history AS (
                SELECT
                    identificador,
                    admission_ts,
                    discharge_ts,

                    LAG(discharge_ts) OVER (
                        PARTITION BY identificador
                        ORDER BY admission_ts
                    ) AS previous_discharge_ts

                FROM hospitalization_target

                WHERE eligible_for_readmission_model = 1
            )

            SELECT
                COUNT(*) AS negative_episodes,
                COUNT(DISTINCT identificador) AS patients
            FROM history
            WHERE previous_discharge_ts IS NOT NULL
              AND DATEDIFF(
                    'day',
                    previous_discharge_ts,
                    admission_ts
                  ) < 0
        """).fetchdf()
    )

    print("\nNegative examples:")
    print(
        conn.execute("""
            WITH history AS (
                SELECT
                    identificador,
                    admission_ts,
                    discharge_ts,

                    LAG(discharge_ts) OVER (
                        PARTITION BY identificador
                        ORDER BY admission_ts
                    ) AS previous_discharge_ts

                FROM hospitalization_target

                WHERE eligible_for_readmission_model = 1
            )

            SELECT
                identificador,
                admission_ts,
                previous_discharge_ts,

                DATEDIFF(
                    'day',
                    previous_discharge_ts,
                    admission_ts
                ) AS days_since_previous_hospitalization

            FROM history

            WHERE previous_discharge_ts IS NOT NULL
              AND DATEDIFF(
                    'day',
                    previous_discharge_ts,
                    admission_ts
                  ) < 0

            ORDER BY days_since_previous_hospitalization

            LIMIT 30
        """).fetchdf()
    )

finally:
    conn.close()