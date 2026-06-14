import duckdb

conn = duckdb.connect("data/hospital_readmissions.duckdb")

print(
    conn.execute("""
        SELECT
            days_until_next_admission,
            COUNT(*) AS episodes
        FROM hospitalization_target
        WHERE days_until_next_admission BETWEEN 0 AND 30
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()
)

conn.close()