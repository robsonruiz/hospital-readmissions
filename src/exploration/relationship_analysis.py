import duckdb
from itertools import combinations

TABLES = [
    "appointments",
    "regulated_exams",
    "non_regulated_exams",
    "requests",
    "hospitalizations",
    "billing",
]

conn = duckdb.connect("data/hospital_readmissions.duckdb")

for table_a, table_b in combinations(TABLES, 2):

    try:
        overlap = conn.execute(f"""
            SELECT COUNT(DISTINCT a.identificador)
            FROM {table_a} a
            INNER JOIN {table_b} b
                ON a.identificador = b.identificador
        """).fetchone()[0]

        print(
            f"{table_a:<25} ↔ {table_b:<25} : {overlap:,}"
        )

    except Exception as e:
        print(f"Error comparing {table_a} and {table_b}: {e}")

conn.close()