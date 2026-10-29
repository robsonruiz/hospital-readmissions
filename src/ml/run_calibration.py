from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.prepare_data import (
    load_ml_data,
    prepare_data,
)
from src.ml.calibration_analysis import (
    run_calibration_analysis,
)


def main():
    print("\n" + "=" * 70)
    print("MODEL CALIBRATION")
    print("=" * 70)

    print("\n=== Loading ML data ===")

    df = load_ml_data()

    print(
        f"Loaded: {len(df):,} records"
    )

    print(
        "\n=== Creating temporal split ==="
    )

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        train_df,
        validation_df,
        test_df,
        train_cutoff,
        validation_cutoff,
    ) = prepare_data(df)

    print(
        f"\nX_train: {X_train.shape}"
    )

    print(
        f"X_valid: {X_valid.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        "\n=== Calibration ==="
    )

    run_calibration_analysis(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "CALIBRATION COMPLETED"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()