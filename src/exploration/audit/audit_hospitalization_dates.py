from pathlib import Path

import duckdb
import pandas as pd


DATABASE_PATH = "data/hospital_readmissions.duckdb"
OUTPUT_DIR = Path("reports/audit/dates")


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def audit_hospitalization_dates() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(DATABASE_PATH)

    # ==========================================================
    # 1. General date and LOS quality
    # ==========================================================

    print_section("1. GENERAL DATE AND LENGTH OF STAY QUALITY")

    general_summary = conn.execute("""
        SELECT
            COUNT(*) AS total_episodes,

            SUM(
                CASE
                    WHEN admission_ts IS NULL THEN 1
                    ELSE 0
                END
            ) AS missing_admission,

            SUM(
                CASE
                    WHEN discharge_ts IS NULL THEN 1
                    ELSE 0
                END
            ) AS missing_discharge,

            SUM(
                CASE
                    WHEN discharge_ts < admission_ts THEN 1
                    ELSE 0
                END
            ) AS negative_los,

            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) = 0
                    THEN 1
                    ELSE 0
                END
            ) AS zero_day_los,

            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 30
                    THEN 1
                    ELSE 0
                END
            ) AS los_over_30_days,

            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 90
                    THEN 1
                    ELSE 0
                END
            ) AS los_over_90_days,

            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 180
                    THEN 1
                    ELSE 0
                END
            ) AS los_over_180_days,

            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 365
                    THEN 1
                    ELSE 0
                END
            ) AS los_over_365_days,

            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 1000
                    THEN 1
                    ELSE 0
                END
            ) AS los_over_1000_days,

            MIN(
                DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                )
            ) AS minimum_los,

            MAX(
                DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                )
            ) AS maximum_los,

            ROUND(
                AVG(
                    DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    )
                ),
                2
            ) AS average_los

        FROM hospitalization_target
    """).fetchdf()

    print(general_summary)

    # ==========================================================
    # 2. Admission and discharge year ranges
    # ==========================================================

    print_section("2. ADMISSION AND DISCHARGE YEAR RANGES")

    year_ranges = conn.execute("""
        SELECT
            MIN(admission_ts) AS earliest_admission,
            MAX(admission_ts) AS latest_admission,
            MIN(discharge_ts) AS earliest_discharge,
            MAX(discharge_ts) AS latest_discharge,

            MIN(YEAR(admission_ts)) AS minimum_admission_year,
            MAX(YEAR(admission_ts)) AS maximum_admission_year,

            MIN(YEAR(discharge_ts)) AS minimum_discharge_year,
            MAX(YEAR(discharge_ts)) AS maximum_discharge_year

        FROM hospitalization_target
    """).fetchdf()

    print(year_ranges)

    # ==========================================================
    # 3. Episodes outside expected date range
    # ==========================================================

    print_section("3. EPISODES OUTSIDE EXPECTED DATE RANGE")

    out_of_range = conn.execute("""
        SELECT
            identificador,
            admission_ts,
            discharge_ts,

            DATEDIFF(
                'day',
                admission_ts,
                discharge_ts
            ) AS length_of_stay_days,

            executante,
            especialidade,
            tipoleito,
            codigocid,
            motivoalta

        FROM hospitalization_target

        WHERE
            admission_ts < TIMESTAMP '2023-01-01'
            OR admission_ts >= TIMESTAMP '2027-01-01'
            OR discharge_ts < TIMESTAMP '2023-01-01'
            OR discharge_ts >= TIMESTAMP '2027-01-01'

        ORDER BY
            admission_ts,
            discharge_ts
    """).fetchdf()

    print(out_of_range.head(100))

    out_of_range.to_csv(
        OUTPUT_DIR / "episodes_outside_expected_range.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================================
    # 4. Longest hospital stays
    # ==========================================================

    print_section("4. LONGEST HOSPITAL STAYS")

    longest_stays = conn.execute("""
        SELECT
            identificador,
            admission_ts,
            discharge_ts,

            DATEDIFF(
                'day',
                admission_ts,
                discharge_ts
            ) AS length_of_stay_days,

            executante,
            municipioexecutante,
            especialidade,
            tipoleito,
            carater,
            codigocid,
            motivoalta

        FROM hospitalization_target

        WHERE
            admission_ts IS NOT NULL
            AND discharge_ts IS NOT NULL

        ORDER BY
            length_of_stay_days DESC

        LIMIT 100
    """).fetchdf()

    print(longest_stays.head(50))

    longest_stays.to_csv(
        OUTPUT_DIR / "longest_hospital_stays.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================================
    # 5. Long stays by hospital
    # ==========================================================

    print_section("5. LONG STAYS BY HOSPITAL")

    long_stays_by_hospital = conn.execute("""
        SELECT
            executante,

            COUNT(*) AS total_episodes,

            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 90
                    THEN 1
                    ELSE 0
                END
            ) AS los_over_90_days,

            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 180
                    THEN 1
                    ELSE 0
                END
            ) AS los_over_180_days,

            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 365
                    THEN 1
                    ELSE 0
                END
            ) AS los_over_365_days,

            ROUND(
                AVG(
                    DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    )
                ),
                2
            ) AS average_los,

            MAX(
                DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                )
            ) AS maximum_los

        FROM hospitalization_target

        WHERE
            admission_ts IS NOT NULL
            AND discharge_ts IS NOT NULL

        GROUP BY executante

        HAVING
            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 90
                    THEN 1
                    ELSE 0
                END
            ) > 0

        ORDER BY los_over_365_days DESC, los_over_90_days DESC
    """).fetchdf()

    print(long_stays_by_hospital)

    long_stays_by_hospital.to_csv(
        OUTPUT_DIR / "long_stays_by_hospital.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================================
    # 6. Long stays by specialty
    # ==========================================================

    print_section("6. LONG STAYS BY SPECIALTY")

    long_stays_by_specialty = conn.execute("""
        SELECT
            especialidade,

            COUNT(*) AS total_episodes,

            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 90
                    THEN 1
                    ELSE 0
                END
            ) AS los_over_90_days,

            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 180
                    THEN 1
                    ELSE 0
                END
            ) AS los_over_180_days,

            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 365
                    THEN 1
                    ELSE 0
                END
            ) AS los_over_365_days,

            ROUND(
                AVG(
                    DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    )
                ),
                2
            ) AS average_los,

            MAX(
                DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                )
            ) AS maximum_los

        FROM hospitalization_target

        WHERE
            admission_ts IS NOT NULL
            AND discharge_ts IS NOT NULL

        GROUP BY especialidade

        HAVING
            SUM(
                CASE
                    WHEN DATEDIFF(
                        'day',
                        admission_ts,
                        discharge_ts
                    ) > 90
                    THEN 1
                    ELSE 0
                END
            ) > 0

        ORDER BY los_over_365_days DESC, los_over_90_days DESC
    """).fetchdf()

    print(long_stays_by_specialty)

    long_stays_by_specialty.to_csv(
        OUTPUT_DIR / "long_stays_by_specialty.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================================
    # 7. LOS distribution buckets
    # ==========================================================

    print_section("7. LENGTH OF STAY DISTRIBUTION")

    los_distribution = conn.execute("""
        SELECT
            CASE
                WHEN discharge_ts IS NULL
                    THEN 'Missing discharge'

                WHEN DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                ) < 0
                    THEN 'Negative'

                WHEN DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                ) = 0
                    THEN '0 days'

                WHEN DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                ) <= 1
                    THEN '1 day'

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

                WHEN DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                ) <= 30
                    THEN '15-30 days'

                WHEN DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                ) <= 90
                    THEN '31-90 days'

                WHEN DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                ) <= 180
                    THEN '91-180 days'

                WHEN DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                ) <= 365
                    THEN '181-365 days'

                WHEN DATEDIFF(
                    'day',
                    admission_ts,
                    discharge_ts
                ) <= 1000
                    THEN '366-1000 days'

                ELSE '1000+ days'
            END AS los_bucket,

            COUNT(*) AS episodes

        FROM hospitalization_target

        GROUP BY los_bucket

        ORDER BY
            CASE los_bucket
                WHEN 'Negative' THEN 1
                WHEN '0 days' THEN 2
                WHEN '1 day' THEN 3
                WHEN '2-3 days' THEN 4
                WHEN '4-7 days' THEN 5
                WHEN '8-14 days' THEN 6
                WHEN '15-30 days' THEN 7
                WHEN '31-90 days' THEN 8
                WHEN '91-180 days' THEN 9
                WHEN '181-365 days' THEN 10
                WHEN '366-1000 days' THEN 11
                WHEN '1000+ days' THEN 12
                WHEN 'Missing discharge' THEN 13
            END
    """).fetchdf()

    print(los_distribution)

    # ==========================================================
    # 8. Zero-day stays
    # ==========================================================

    print_section("8. ZERO-DAY STAYS")

    zero_day_summary = conn.execute("""
        SELECT
            COUNT(*) AS zero_day_episodes,
            COUNT(DISTINCT identificador) AS affected_patients,

            ROUND(
                COUNT(*) * 100.0
                / (
                    SELECT COUNT(*)
                    FROM hospitalization_target
                ),
                2
            ) AS percentage_of_episodes

        FROM hospitalization_target

        WHERE
            discharge_ts IS NOT NULL
            AND DATEDIFF(
                'day',
                admission_ts,
                discharge_ts
            ) = 0
    """).fetchdf()

    print(zero_day_summary)

    zero_day_by_hospital = conn.execute("""
        SELECT
            executante,
            COUNT(*) AS zero_day_episodes

        FROM hospitalization_target

        WHERE
            discharge_ts IS NOT NULL
            AND DATEDIFF(
                'day',
                admission_ts,
                discharge_ts
            ) = 0

        GROUP BY executante
        ORDER BY zero_day_episodes DESC
        LIMIT 30
    """).fetchdf()

    print("\nZero-day stays by hospital:")
    print(zero_day_by_hospital)

    zero_day_by_specialty = conn.execute("""
        SELECT
            especialidade,
            COUNT(*) AS zero_day_episodes

        FROM hospitalization_target

        WHERE
            discharge_ts IS NOT NULL
            AND DATEDIFF(
                'day',
                admission_ts,
                discharge_ts
            ) = 0

        GROUP BY especialidade
        ORDER BY zero_day_episodes DESC
        LIMIT 30
    """).fetchdf()

    print("\nZero-day stays by specialty:")
    print(zero_day_by_specialty)

    # ==========================================================
    # 9. Suspected year-entry errors
    # ==========================================================

    print_section("9. SUSPECTED YEAR-ENTRY ERRORS")

    suspected_year_errors = conn.execute("""
        SELECT
            identificador,
            admission_ts,
            discharge_ts,

            YEAR(admission_ts) AS admission_year,
            YEAR(discharge_ts) AS discharge_year,

            DATEDIFF(
                'day',
                admission_ts,
                discharge_ts
            ) AS length_of_stay_days,

            executante,
            especialidade,
            codigocid,
            motivoalta

        FROM hospitalization_target

        WHERE
            ABS(
                YEAR(discharge_ts)
                - YEAR(admission_ts)
            ) >= 2

        ORDER BY length_of_stay_days DESC
    """).fetchdf()

    print(suspected_year_errors.head(100))

    suspected_year_errors.to_csv(
        OUTPUT_DIR / "suspected_year_entry_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================================
    # 10. Extreme LOS and next admission
    # ==========================================================

    print_section("10. EXTREME LOS AND NEXT ADMISSION")

    extreme_los_next_admission = conn.execute("""
        SELECT
            identificador,
            admission_ts,
            discharge_ts,
            next_admission_ts,

            DATEDIFF(
                'day',
                admission_ts,
                discharge_ts
            ) AS length_of_stay_days,

            days_until_next_admission,
            executante,
            especialidade,
            codigocid,
            motivoalta,
            readmitted_30d_clean

        FROM hospitalization_target

        WHERE
            DATEDIFF(
                'day',
                admission_ts,
                discharge_ts
            ) > 365

        ORDER BY length_of_stay_days DESC
    """).fetchdf()

    print(extreme_los_next_admission.head(100))

    extreme_los_next_admission.to_csv(
        OUTPUT_DIR / "extreme_los_with_next_admission.csv",
        index=False,
        encoding="utf-8-sig"
    )

    conn.close()

    print("\nAudit completed.")
    print(f"CSV files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    audit_hospitalization_dates()