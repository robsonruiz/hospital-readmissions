import duckdb
import pandas as pd


DATABASE_PATH = "data/hospital_readmissions.duckdb"

PATIENT_ID = (
    "504ac34ff10ba0c69bd09d503dfe942d7576b8a843f015"
)


def inspect_extreme_los() -> None:
    conn = duckdb.connect(DATABASE_PATH)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 250)

    print("\n=== RAW HOSPITALIZATION RECORDS ===")

    raw_records = conn.execute(
        """
        SELECT
            identificador,
            datasolicitacao,
            datahorasolicitacao,
            datahorainternacao,
            datahoraalta,
            situacao,
            executante,
            municipioexecutante,
            especialidade,
            tipoleito,
            carater,
            codigocid,
            motivoalta
        FROM hospitalizations
        WHERE identificador LIKE ?
        ORDER BY
            datahorasolicitacao,
            datasolicitacao
        """,
        [f"{PATIENT_ID}%"]
    ).fetchdf()

    print(raw_records)

    print("\n=== CONSOLIDATED EPISODES ===")

    episodes = conn.execute(
        """
        SELECT
            identificador,
            datahorainternacao,
            datahoraalta,
            last_update,
            admission_ts,
            discharge_ts,
            DATEDIFF(
                'day',
                admission_ts,
                discharge_ts
            ) AS length_of_stay_days,
            executante,
            especialidade,
            codigocid,
            motivoalta
        FROM hospitalization_target
        WHERE identificador LIKE ?
        ORDER BY admission_ts
        """,
        [f"{PATIENT_ID}%"]
    ).fetchdf()

    print(episodes)

    conn.close()


if __name__ == "__main__":
    inspect_extreme_los()