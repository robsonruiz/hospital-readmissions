from pathlib import Path
import duckdb
import pandas as pd

DATABASE_PATH = "data/hospital_readmissions.duckdb"

TABLES = [
    "appointments",
    "regulated_exams",
    "non_regulated_exams",
    "requests",
    "hospitalizations",
    "billing",
]

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

conn = duckdb.connect(DATABASE_PATH)

catalog_rows = []

for table in TABLES:

    print(f"\n=== {table.upper()} ===")

    schema = conn.execute(f"""
        DESCRIBE {table}
    """).fetchdf()

    print(schema)

    total_rows = conn.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    for _, row in schema.iterrows():

        column_name = row["column_name"]

        try:
            null_count = conn.execute(f"""
                SELECT COUNT(*)
                FROM {table}
                WHERE "{column_name}" IS NULL
            """).fetchone()[0]

            distinct_count = conn.execute(f"""
                SELECT COUNT(DISTINCT "{column_name}")
                FROM {table}
            """).fetchone()[0]

        except:
            null_count = None
            distinct_count = None

        catalog_rows.append({
            "table": table,
            "column": column_name,
            "type": row["column_type"],
            "rows": total_rows,
            "nulls": null_count,
            "distinct_values": distinct_count
        })

catalog = pd.DataFrame(catalog_rows)

catalog.to_csv(
    OUTPUT_DIR / "data_catalog.csv",
    index=False
)

print("\nCatalog saved.")