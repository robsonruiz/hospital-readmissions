import duckdb

conn = duckdb.connect("data/hospital_readmissions.duckdb")

print(
    conn.execute("""
        SELECT
            motivoalta,
            COUNT(*) AS episodes
        FROM hospitalization_target
        WHERE days_until_next_admission = 0
        GROUP BY motivoalta
        ORDER BY episodes DESC
    """).fetchdf()
)

print(
    conn.execute("""
        SELECT
            executante,
            COUNT(*) AS episodes
        FROM hospitalizations
        WHERE identificador IN (
            SELECT identificador
            FROM hospitalization_target
            WHERE days_until_next_admission = 0
        )
        GROUP BY executante
        ORDER BY episodes DESC
        LIMIT 20
    """).fetchdf()
)

conn.close()