from src.ingestion.load_raw_data import load_raw_data
from src.ingestion.export_parquets import export_parquets

from src.features.build_hospitalization_episodes import (
    build_hospitalization_episodes,
)
from src.features.build_readmission_target import (
    build_readmission_target,
)
from src.features.build_analytics_dataset import (
    build_analytics_dataset,
)

def main() -> None:
    print("\n=== STEP 1 - Loading CSV files ===")
    load_raw_data()

    print("\n=== STEP 2 - Exporting Parquet files ===")
    export_parquets()

    print("\n=== STEP 3 - Building Hospitalization Episodes ===")
    build_hospitalization_episodes()

    print("\n=== STEP 4 - Building Unplanned Readmission Target ===")
    build_readmission_target()

    print("\n=== STEP 5 - Building Analytics Dataset ===")
    build_analytics_dataset()

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()