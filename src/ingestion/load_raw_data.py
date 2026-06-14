from pathlib import Path
import duckdb


def load_raw_data():

    RAW_DATA_DIR = Path("data/raw")
    DATABASE_PATH = "data/hospital_readmissions.duckdb"

    DATASETS = {
        "appointments": "1_consultas_gercon.csv",
        "regulated_exams": "2_exames_regulados_gercon.csv",
        "non_regulated_exams": "3_exames_nao_regulados_gercon.csv",
        "requests": "4_solicitacoes_gerpac.csv",
        "hospitalizations": "5_internacoes_gerint.csv",
        "billing": "6_faturamento_gerint.csv",
    }

    conn = duckdb.connect(DATABASE_PATH)

    for table_name, file_name in DATASETS.items():

        file_path = RAW_DATA_DIR / file_name

        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue

        print(f"\nLoading {file_name}...")

        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_csv_auto(
                '{file_path.as_posix()}',
                header=True,
                ignore_errors=True
            )
        """)

        row_count = conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]

        print(f"✓ {table_name}: {row_count:,} rows")

    conn.close()

    print("\nData ingestion completed successfully.")


if __name__ == "__main__":
    load_raw_data()