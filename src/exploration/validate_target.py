import duckdb


DATABASE_PATH = "data/hospital_readmissions.duckdb"


def validate_readmission_target():
    conn = duckdb.connect(DATABASE_PATH)

    print("\n=== Target distribution ===")
    print(
        conn.execute("""
            SELECT
                readmitted_30d_clean,
                COUNT(*) AS episodes,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
            FROM hospitalization_target
            GROUP BY readmitted_30d_clean
            ORDER BY readmitted_30d_clean
        """).fetchdf()
    )

    print("\n=== Raw vs clean target ===")
    print(
        conn.execute("""
            SELECT
                SUM(readmitted_30d_raw) AS raw_readmissions,
                SUM(readmitted_30d_clean) AS clean_readmissions,
                COUNT(*) AS total_episodes,
                ROUND(SUM(readmitted_30d_raw) * 100.0 / COUNT(*), 2) AS raw_rate,
                ROUND(SUM(readmitted_30d_clean) * 100.0 / COUNT(*), 2) AS clean_rate
            FROM hospitalization_target
        """).fetchdf()
    )

    print("\n=== Excluded discharge reasons ===")
    print(
        conn.execute("""
            SELECT
                motivoalta,
                COUNT(*) AS episodes,
                SUM(readmitted_30d_raw) AS raw_readmissions,
                SUM(readmitted_30d_clean) AS clean_readmissions
            FROM hospitalization_target
            WHERE motivoalta IN (
                'Transferência',
                'Óbito',
                'Encerramento administrativo'
            )
            GROUP BY motivoalta
            ORDER BY episodes DESC
        """).fetchdf()
    )

    print("\n=== Days until next admission ===")
    print(
        conn.execute("""
            SELECT
                days_until_next_admission,
                COUNT(*) AS episodes
            FROM hospitalization_target
            WHERE days_until_next_admission BETWEEN 0 AND 30
            GROUP BY days_until_next_admission
            ORDER BY days_until_next_admission
        """).fetchdf()
    )

    conn.close()


if __name__ == "__main__":
    validate_readmission_target()