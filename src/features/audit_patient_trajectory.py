from pathlib import Path
import re

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "data" / "hospital_readmissions.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "models" / "trajectory_audit"

CHUNK_SIZE = 100_000

FILES = {
    "consultas": DATA_DIR / "1_consultas_gercon.csv",
    "exames_regulados": DATA_DIR / "2_exames_regulados_gercon.csv",
    "exames_nao_regulados": DATA_DIR / "3_exames_nao_regulados_gercon.csv",
    "solicitacoes": DATA_DIR / "4_solicitacoes_gerpac.csv",
    "internacoes": DATA_DIR / "5_internacoes_gerint.csv",
}


def read_csv_header(path):
    """
    Read only the header of a CSV.
    Handles the semicolon delimiter used by the files.
    """
    return pd.read_csv(
        path,
        sep=";",
        nrows=0,
        encoding="utf-8",
    ).columns.tolist()


def normalize_identifier(value):
    """
    Normalize patient identifiers.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def audit_file_basic(name, path):
    """
    Basic audit using chunks.
    """
    print()
    print("=" * 80)
    print(f"AUDITING: {name}")
    print("=" * 80)

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        return None

    print(f"File: {path}")

    columns = read_csv_header(path)

    print()
    print("Columns:")
    for column in columns:
        print(f"  - {column}")

    total_rows = 0
    valid_identifiers = 0
    unique_identifiers = set()

    status_counts = {}
    unit_counts = {}
    executor_counts = {}

    for chunk in pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        total_rows += len(chunk)

        if "identificador" in chunk.columns:
            identifiers = (
                chunk["identificador"]
                .map(normalize_identifier)
            )

            valid = identifiers.dropna()

            valid_identifiers += len(valid)
            unique_identifiers.update(
                valid.unique()
            )

        for column in [
            "situacao",
            "status",
        ]:
            if column in chunk.columns:
                counts = (
                    chunk[column]
                    .fillna("MISSING")
                    .astype(str)
                    .value_counts()
                )

                for value, count in counts.items():
                    status_counts[value] = (
                        status_counts.get(value, 0)
                        + int(count)
                    )

        for column in [
            "unidade_solicitante",
            "unidadesolicitante",
        ]:
            if column in chunk.columns:
                counts = (
                    chunk[column]
                    .fillna("MISSING")
                    .astype(str)
                    .value_counts()
                )

                for value, count in counts.items():
                    unit_counts[value] = (
                        unit_counts.get(value, 0)
                        + int(count)
                    )

        for column in [
            "unidade_executante",
            "executante",
        ]:
            if column in chunk.columns:
                counts = (
                    chunk[column]
                    .fillna("MISSING")
                    .astype(str)
                    .value_counts()
                )

                for value, count in counts.items():
                    executor_counts[value] = (
                        executor_counts.get(value, 0)
                        + int(count)
                    )

    result = {
        "source": name,
        "rows": total_rows,
        "rows_with_identifier": valid_identifiers,
        "unique_patients": len(unique_identifiers),
    }

    print()
    print(f"Rows: {total_rows:,}")
    print(
        f"Rows with identifier: "
        f"{valid_identifiers:,}"
    )
    print(
        f"Unique patients: "
        f"{len(unique_identifiers):,}"
    )

    if status_counts:
        print()
        print("Status:")
        for value, count in sorted(
            status_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:20]:
            print(
                f"  {value}: {count:,}"
            )

    if unit_counts:
        print()
        print("Top requesting units:")

        for value, count in sorted(
            unit_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:30]:
            print(
                f"  {value}: {count:,}"
            )

    if executor_counts:
        print()
        print("Top executing units:")

        for value, count in sorted(
            executor_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:30]:
            print(
                f"  {value}: {count:,}"
            )

    return result


def load_hospitalization_patients():
    """
    Load the patient identifiers present in hospitalization_episodes.
    """
    print()
    print("=" * 80)
    print("LOADING HOSPITALIZATION PATIENTS")
    print("=" * 80)

    con = duckdb.connect(
        DB_PATH,
        read_only=True,
    )

    df = con.execute(
        """
        SELECT DISTINCT identificador
        FROM hospitalization_episodes
        WHERE identificador IS NOT NULL
        """
    ).df()

    con.close()

    patients = set(
        df["identificador"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    print(
        f"Hospitalization patients: "
        f"{len(patients):,}"
    )

    return patients


def get_patient_ids_from_csv(path):
    """
    Extract unique patient identifiers from a CSV
    without loading the complete file.
    """
    patients = set()

    for chunk in pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        usecols=["identificador"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        identifiers = (
            chunk["identificador"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        patients.update(
            identifiers.unique()
        )

    return patients


def audit_patient_overlap(hospitalization_patients):
    """
    Measure patient overlap between hospitalization data
    and each longitudinal source.
    """
    print()
    print("=" * 80)
    print("PATIENT OVERLAP")
    print("=" * 80)

    rows = []

    for name, path in FILES.items():
        if not path.exists():
            continue

        if name == "internacoes":
            continue

        print()
        print(
            f"Processing {name}..."
        )

        patients = get_patient_ids_from_csv(
            path
        )

        intersection = (
            patients
            & hospitalization_patients
        )

        rows.append(
            {
                "source": name,
                "unique_patients": len(
                    patients
                ),
                "patients_also_in_hospitalizations": len(
                    intersection
                ),
                "coverage_among_hospitalization_patients_pct": (
                    len(intersection)
                    / len(hospitalization_patients)
                    * 100
                    if hospitalization_patients
                    else 0
                ),
            }
        )

        print(
            f"  Unique patients: "
            f"{len(patients):,}"
        )

        print(
            f"  Also hospitalized: "
            f"{len(intersection):,}"
        )

        print(
            f"  Coverage: "
            f"{len(intersection) / len(hospitalization_patients) * 100:.2f}%"
        )

    return pd.DataFrame(rows)


def classify_primary_care_unit(value):
    """
    Heuristic classification of primary-care units.

    This is an audit heuristic only. It does not modify the
    modeling dataset.
    """
    if pd.isna(value):
        return False

    text = str(value).upper()

    patterns = [
        r"\bUBS\b",
        r"UNIDADE DE SAUDE",
        r"UNIDADE DE SAÚDE",
        r"CLINICA DA FAMILIA",
        r"CLÍNICA DA FAMÍLIA",
        r"ESF",
        r"ESTRATEGIA DE SAUDE DA FAMILIA",
        r"ESTRATÉGIA DE SAÚDE DA FAMÍLIA",
        r"SAUDE DA FAMILIA",
        r"SAÚDE DA FAMÍLIA",
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def audit_primary_care(name, path):
    """
    Identify units that appear to represent primary care.
    """
    print()
    print("=" * 80)
    print(f"PRIMARY CARE AUDIT: {name}")
    print("=" * 80)

    unit_columns = [
        "unidade_solicitante",
        "unidadesolicitante",
        "unidade_executante",
        "executante",
    ]

    available_columns = read_csv_header(
        path
    )

    available = [
        column
        for column in unit_columns
        if column in available_columns
    ]

    if not available:
        print(
            "No unit column found."
        )
        return pd.DataFrame()

    counts = {}

    for chunk in pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        usecols=available,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        for column in available:
            values = (
                chunk[column]
                .dropna()
                .astype(str)
                .str.strip()
            )

            for value in values:
                if classify_primary_care_unit(
                    value
                ):
                    counts[value] = (
                        counts.get(value, 0)
                        + 1
                    )

    result = pd.DataFrame(
        [
            {
                "source": name,
                "unit": unit,
                "events": count,
            }
            for unit, count in sorted(
                counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
    )

    print()
    print(
        f"Primary-care-like units found: "
        f"{len(result):,}"
    )

    if not result.empty:
        print()
        print(
            result.head(50).to_string(
                index=False
            )
        )

    return result


def main():
    print()
    print("=" * 80)
    print("PATIENT TRAJECTORY AUDIT")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 1. Basic audit
    # ------------------------------------------------------------------
    summaries = []

    for name, path in FILES.items():
        result = audit_file_basic(
            name,
            path,
        )

        if result is not None:
            summaries.append(result)

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        OUTPUT_DIR
        / "01_source_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # 2. Hospitalization patient universe
    # ------------------------------------------------------------------
    hospitalization_patients = (
        load_hospitalization_patients()
    )

    # ------------------------------------------------------------------
    # 3. Patient overlap
    # ------------------------------------------------------------------
    overlap_df = audit_patient_overlap(
        hospitalization_patients
    )

    overlap_path = (
        OUTPUT_DIR
        / "02_patient_overlap.csv"
    )

    overlap_df.to_csv(
        overlap_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # 4. Primary care units
    # ------------------------------------------------------------------
    primary_care_results = []

    for name in [
        "consultas",
        "exames_regulados",
        "exames_nao_regulados",
        "solicitacoes",
    ]:
        path = FILES[name]

        if not path.exists():
            continue

        result = audit_primary_care(
            name,
            path,
        )

        if not result.empty:
            primary_care_results.append(
                result
            )

    if primary_care_results:
        primary_care_df = pd.concat(
            primary_care_results,
            ignore_index=True,
        )
    else:
        primary_care_df = pd.DataFrame()

    primary_care_path = (
        OUTPUT_DIR
        / "03_primary_care_units.csv"
    )

    primary_care_df.to_csv(
        primary_care_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # 5. Final summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("AUDIT COMPLETED")
    print("=" * 80)

    print()
    print("Files generated:")

    print(
        f"  {summary_path}"
    )

    print(
        f"  {overlap_path}"
    )

    print(
        f"  {primary_care_path}"
    )

    print()
    print(
        "No modeling dataset was modified."
    )

    print(
        "No ML model was retrained."
    )

    print(
        "No TEST data was used for modeling."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()