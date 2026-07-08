import duckdb


DATABASE_PATH = "data/hospital_readmissions.duckdb"


def analyze_administrative_closure():
    conn = duckdb.connect(DATABASE_PATH)

    print("\n=== Administrative Closure by Status ===")
    print(
        conn.execute("""
            SELECT
                motivoalta,
                situacao,
                COUNT(*) AS episodes
            FROM hospitalizations
            WHERE motivoalta = 'Encerramento administrativo'
            GROUP BY motivoalta, situacao
            ORDER BY episodes DESC
        """).fetchdf()
    )

    print("\n=== Administrative Closure by Hospital ===")
    print(
        conn.execute("""
            SELECT
                executante,
                COUNT(*) AS episodes
            FROM hospitalizations
            WHERE motivoalta = 'Encerramento administrativo'
            GROUP BY executante
            ORDER BY episodes DESC
        """).fetchdf()
    )

    print("\n=== Administrative Closure Dates ===")
    print(
        conn.execute("""
            SELECT
                MIN(datasolicitacao) AS min_request_date,
                MAX(datasolicitacao) AS max_request_date,
                COUNT(*) AS episodes
            FROM hospitalizations
            WHERE motivoalta = 'Encerramento administrativo'
        """).fetchdf()
    )

    print("\n=== Administrative Closure with Next Admission ===")
    print(
        conn.execute("""
            SELECT
                COUNT(*) AS total_closures,
                SUM(
                    CASE
                        WHEN days_until_next_admission BETWEEN 0 AND 30
                        THEN 1
                        ELSE 0
                    END
                ) AS next_admission_0_30d,
                SUM(
                    CASE
                        WHEN days_until_next_admission BETWEEN 2 AND 30
                        THEN 1
                        ELSE 0
                    END
                ) AS next_admission_2_30d
            FROM hospitalization_target
            WHERE motivoalta = 'Encerramento administrativo'
        """).fetchdf()
    )

    print("\n=== Sample Administrative Closure Records ===")
    print(
        conn.execute("""
            SELECT
                identificador,
                datasolicitacao,
                datahorasolicitacao,
                datahorainternacao,
                datahoraalta,
                situacao,
                executante,
                especialidade,
                tipoleito,
                carater,
                codigocid,
                motivoalta
            FROM hospitalizations
            WHERE motivoalta = 'Encerramento administrativo'
            ORDER BY datasolicitacao
            LIMIT 30
        """).fetchdf()
    )

    conn.close()


if __name__ == "__main__":
    analyze_administrative_closure()