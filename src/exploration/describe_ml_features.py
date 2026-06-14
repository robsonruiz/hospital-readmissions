import duckdb

conn = duckdb.connect(
    "data/hospital_readmissions.duckdb"
)

print(
    conn.execute("""
        DESCRIBE ml_features
    """).fetchdf()
)

conn.close()