import duckdb

conn = duckdb.connect("data/hospital_readmissions.duckdb")

print(
    conn.execute("""
        SELECT
            MIN(length_of_stay_days),
            MAX(length_of_stay_days),
            AVG(length_of_stay_days)
        FROM ml_features
    """).fetchdf()
)

print(
    conn.execute("""
        SELECT
            previous_hospitalizations,
            COUNT(*)
        FROM ml_features
        GROUP BY 1
        ORDER BY 1
        LIMIT 20
    """).fetchdf()
)

conn.close()