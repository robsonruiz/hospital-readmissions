from pathlib import Path
import duckdb

def export_parquets():

    DATABASE_PATH = "data/hospital_readmissions.duckdb"

    TABLES = [
        "appointments",
        "regulated_exams",
        "non_regulated_exams",
        "requests",
        "hospitalizations",
        "billing",
    ]

    OUTPUT_DIR = Path("data/processed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(DATABASE_PATH)

    for table_name in TABLES:

        output_file = OUTPUT_DIR / f"{table_name}.parquet"

        print(f"Exporting {table_name}...")

        conn.execute(f"""
            COPY {table_name}
            TO '{output_file.as_posix()}'
            (FORMAT PARQUET);
        """)

        print(f"✓ Saved: {output_file}")

    conn.close()

    print("\nParquet export completed.")