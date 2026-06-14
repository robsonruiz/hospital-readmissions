import duckdb

TABLES = [
    "appointments",
    "regulated_exams",
    "non_regulated_exams",
    "requests",
    "hospitalizations",
    "billing",
]

conn = duckdb.connect("data/healthcare_data.duckdb")

for table in TABLES:

    print("\n")
    print("=" * 80)
    print(table.upper())

    df = conn.execute(f"""
        SELECT *
        FROM {table}
        LIMIT 5
    """).df()

    print(df)