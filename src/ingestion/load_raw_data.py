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
        # seu código
        pass

    conn.close()