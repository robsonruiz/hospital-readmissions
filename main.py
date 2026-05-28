from src.ingestion.load_raw_data import load_raw_data
from src.ingestion.export_parquets import export_parquets

def main():
    print("Step 1 - Loading CSV files")
    load_raw_data()

    print("Step 2 - Exporting Parquet files")
    export_parquets()

    print("Pipeline completed successfully")

if __name__ == "__main__":
    main()