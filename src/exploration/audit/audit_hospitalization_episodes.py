import duckdb


DATABASE_PATH = "data/hospital_readmissions.duckdb"


def run_query(conn, title, query):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(conn.execute(query).fetchdf())


def main():
    conn = duckdb.connect(DATABASE_PATH)

    run_query(conn, "1. Episode counts", """
        SELECT
            COUNT(*) AS total_episodes,
            COUNT(DISTINCT identificador) AS patients,
            ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT identificador), 2) AS avg_episodes_per_patient
        FROM hospitalization_target
    """)

    run_query(conn, "2. Patients with most episodes", """
        SELECT
            identificador,
            COUNT(*) AS episodes
        FROM hospitalization_target
        GROUP BY identificador
        ORDER BY episodes DESC
        LIMIT 30
    """)

    run_query(conn, "3. Duplicate episodes", """
        SELECT
            identificador,
            admission_ts,
            COUNT(*) AS records
        FROM hospitalization_target
        GROUP BY identificador, admission_ts
        HAVING COUNT(*) > 1
        ORDER BY records DESC
        LIMIT 30
    """)

    run_query(conn, "4. Length of stay quality", """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN discharge_ts IS NULL THEN 1 ELSE 0 END) AS missing_discharge,
            SUM(CASE WHEN DATEDIFF('day', admission_ts, discharge_ts) < 0 THEN 1 ELSE 0 END) AS negative_los,
            SUM(CASE WHEN DATEDIFF('day', admission_ts, discharge_ts) = 0 THEN 1 ELSE 0 END) AS zero_day_los,
            SUM(CASE WHEN DATEDIFF('day', admission_ts, discharge_ts) > 365 THEN 1 ELSE 0 END) AS los_over_365,
            MIN(DATEDIFF('day', admission_ts, discharge_ts)) AS min_los,
            MAX(DATEDIFF('day', admission_ts, discharge_ts)) AS max_los,
            ROUND(AVG(DATEDIFF('day', admission_ts, discharge_ts)), 2) AS avg_los
        FROM hospitalization_target
    """)

    run_query(conn, "5. Overlapping episodes", """
        WITH ordered AS (
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
            COUNT(*) AS overlapping_episodes
        FROM ordered
        WHERE discharge_ts IS NOT NULL
          AND next_admission_ts IS NOT NULL
          AND next_admission_ts < discharge_ts
    """)

    run_query(conn, "6. Very short gaps after discharge", """
        SELECT
            days_until_next_admission,
            COUNT(*) AS episodes
        FROM hospitalization_target
        WHERE days_until_next_admission BETWEEN 0 AND 7
        GROUP BY days_until_next_admission
        ORDER BY days_until_next_admission
    """)

    run_query(conn, "7. Death followed by another admission", """
        SELECT
            COUNT(*) AS death_followed_by_admission
        FROM hospitalization_target
        WHERE motivoalta = 'Óbito'
          AND next_admission_ts IS NOT NULL
    """)

    run_query(conn, "8. Transfer followed by another admission", """
        SELECT
            days_until_next_admission,
            COUNT(*) AS episodes
        FROM hospitalization_target
        WHERE motivoalta = 'Transferência'
          AND days_until_next_admission BETWEEN 0 AND 30
        GROUP BY days_until_next_admission
        ORDER BY days_until_next_admission
    """)

    run_query(conn, "9. Readmission target distribution", """
        SELECT
            readmitted_30d_clean,
            COUNT(*) AS episodes,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
        FROM hospitalization_target
        GROUP BY readmitted_30d_clean
        ORDER BY readmitted_30d_clean
    """)

    run_query(conn, "10. Readmission by discharge reason", """
        SELECT
            motivoalta,
            COUNT(*) AS episodes,
            SUM(readmitted_30d_clean) AS readmissions,
            ROUND(AVG(readmitted_30d_clean) * 100, 2) AS readmission_rate
        FROM hospitalization_target
        GROUP BY motivoalta
        ORDER BY episodes DESC
    """)

    run_query(conn, "11. Sample positive readmissions", """
        SELECT
            identificador,
            admission_ts,
            discharge_ts,
            next_admission_ts,
            days_until_next_admission,
            motivoalta,
            executante,
            especialidade,
            codigocid,
            readmitted_30d_clean
        FROM hospitalization_target
        WHERE readmitted_30d_clean = 1
        ORDER BY RANDOM()
        LIMIT 20
    """)

    run_query(conn, "12. Sample patients with many episodes", """
        WITH top_patients AS (
            SELECT identificador
            FROM hospitalization_target
            GROUP BY identificador
            ORDER BY COUNT(*) DESC
            LIMIT 5
        )
        SELECT
            identificador,
            admission_ts,
            discharge_ts,
            next_admission_ts,
            days_until_next_admission,
            motivoalta,
            executante,
            especialidade,
            codigocid,
            readmitted_30d_clean
        FROM hospitalization_target
        WHERE identificador IN (
            SELECT identificador FROM top_patients
        )
        ORDER BY identificador, admission_ts
    """)

    conn.close()


if __name__ == "__main__":
    main()