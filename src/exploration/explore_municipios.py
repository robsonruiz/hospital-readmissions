import duckdb

conn = duckdb.connect(
    "data/hospital_readmissions.duckdb"
)

print(
    conn.execute("""
        SELECT
            municipioresidencia,
            COUNT(*) AS episodes
        FROM ml_features
        GROUP BY municipioresidencia
        ORDER BY episodes DESC
        LIMIT 20
    """).fetchdf()
)

conn.close()