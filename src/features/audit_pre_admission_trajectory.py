from pathlib import Path
import re

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "data" / "hospital_readmissions.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "models" / "trajectory_audit"

CHUNK_SIZE = 100_000


SOURCE_FILES = {
    "consultas": DATA_DIR / "1_consultas_gercon.csv",
    "exames_regulados": DATA_DIR / "2_exames_regulados_gercon.csv",
    "exames_nao_regulados": DATA_DIR / "3_exames_nao_regulados_gercon.csv",
    "solicitacoes": DATA_DIR / "4_solicitacoes_gerpac.csv",
}


def normalize_identifier(series):
    """
    Normalize patient identifiers.
    """
    return (
        series
        .astype("string")
        .str.strip()
    )


def load_hospitalization_episodes():
    """
    Load hospitalization episodes from hospitalization_target.

    hospitalization_target contains the cleaned hospitalization
    episodes and the admission_ts/discharge_ts timestamps used
    by the readmission pipeline.
    """
    print()
    print("=" * 80)
    print("1. LOADING HOSPITALIZATION EPISODES")
    print("=" * 80)

    con = duckdb.connect(
        DB_PATH,
        read_only=True,
    )

    df = con.execute(
        """
        SELECT
            identificador,
            admission_ts,
            discharge_ts
        FROM hospitalization_target
        WHERE identificador IS NOT NULL
          AND admission_ts IS NOT NULL
        ORDER BY admission_ts
        """
    ).df()

    con.close()

    df["identificador"] = normalize_identifier(
        df["identificador"]
    )

    df["admission_ts"] = pd.to_datetime(
        df["admission_ts"],
        errors="coerce",
    )

    df["discharge_ts"] = pd.to_datetime(
        df["discharge_ts"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "identificador",
            "admission_ts",
        ]
    )

    df["episode_id"] = range(
        1,
        len(df) + 1,
    )

    print(
        f"Episodes: {len(df):,}"
    )

    print(
        f"Patients: "
        f"{df['identificador'].nunique():,}"
    )

    print(
        f"Date range: "
        f"{df['admission_ts'].min().date()} "
        f"-> "
        f"{df['admission_ts'].max().date()}"
    )

    return df

def read_consultas():
    """
    Read consultation events from the large CSV.

    The date used for the trajectory is data_hora_realizacao
    when available, falling back to data_realizacao.
    """
    path = SOURCE_FILES["consultas"]

    print()
    print("=" * 80)
    print("2. LOADING CONSULTATION EVENTS")
    print("=" * 80)

    events = []

    usecols = [
        "data_realizacao",
        "data_hora_realizacao",
        "unidade_solicitante",
        "unidade_executante",
        "situacao",
        "descricao_especialidade",
        "tipo_consulta",
        "cbo",
        "cid",
        "identificador",
    ]

    for chunk in pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        identifier = normalize_identifier(
            chunk["identificador"]
        )

        event_datetime = pd.to_datetime(
            chunk["data_hora_realizacao"],
            errors="coerce",
        )

        fallback_date = pd.to_datetime(
            chunk["data_realizacao"],
            errors="coerce",
        )

        event_datetime = event_datetime.fillna(
            fallback_date
        )

        valid = (
            identifier.notna()
            & event_datetime.notna()
        )

        chunk = chunk.loc[
            valid
        ].copy()

        chunk["identificador"] = (
            identifier.loc[valid]
        )

        chunk["event_datetime"] = (
            event_datetime.loc[valid]
        )

        chunk["source"] = "consultas"

        chunk["is_primary_care"] = (
            chunk.apply(
                classify_primary_care_consultation,
                axis=1,
            )
        )

        events.append(
            chunk[
                [
                    "identificador",
                    "event_datetime",
                    "source",
                    "is_primary_care",
                    "unidade_solicitante",
                    "unidade_executante",
                    "situacao",
                    "descricao_especialidade",
                    "tipo_consulta",
                    "cbo",
                    "cid",
                ]
            ]
        )

    if not events:
        return pd.DataFrame()

    df = pd.concat(
        events,
        ignore_index=True,
    )

    print(
        f"Consultation events: "
        f"{len(df):,}"
    )

    print(
        f"Patients: "
        f"{df['identificador'].nunique():,}"
    )

    print(
        f"Primary-care events: "
        f"{int(df['is_primary_care'].sum()):,}"
    )

    return df


def read_exames_regulados():
    print()
    print("=" * 80)
    print("3. LOADING REGULATED EXAM EVENTS")
    print("=" * 80)

    events = []

    usecols = [
        "datasolicitacao",
        "identificador",
        "cid",
        "situacao",
        "grupoexames",
        "exame",
        "executante",
        "datarealizacao",
    ]

    for chunk in pd.read_csv(
        DATA_DIR / "2_exames_regulados_gercon.csv",
        sep=";",
        usecols=usecols,
        chunksize=100_000,
        low_memory=False,
    ):
        chunk["identificador"] = normalize_identifier(
            chunk["identificador"]
        )

        chunk["datarealizacao"] = pd.to_datetime(
            chunk["datarealizacao"],
            errors="coerce",
        )

        chunk["datasolicitacao"] = pd.to_datetime(
            chunk["datasolicitacao"],
            errors="coerce",
        )

        # Preferimos a data de realização.
        # Quando ela não existe, usamos a data de solicitação.
        chunk["event_datetime"] = chunk["datarealizacao"].fillna(
            chunk["datasolicitacao"]
        )

        chunk = chunk[
            chunk["identificador"].notna()
            & chunk["event_datetime"].notna()
        ].copy()

        # O arquivo de exames regulados não possui
        # "unidadesolicitante". Portanto, não é possível
        # classificar atenção primária pela unidade de origem.
        chunk["is_primary_care"] = False

        events.append(
            chunk[
                [
                    "identificador",
                    "event_datetime",
                    "grupoexames",
                    "exame",
                    "executante",
                    "situacao",
                    "cid",
                    "is_primary_care",
                ]
            ]
        )

    if not events:
        print("No regulated exam events found.")
        return pd.DataFrame(
            columns=[
                "identificador",
                "event_datetime",
                "grupoexames",
                "exame",
                "executante",
                "situacao",
                "cid",
                "is_primary_care",
            ]
        )

    df = pd.concat(
        events,
        ignore_index=True,
    )

    df = df.sort_values(
        [
            "identificador",
            "event_datetime",
        ]
    )

    print(
        f"Regulated exam events: "
        f"{len(df):,}"
    )

    print(
        f"Unique patients: "
        f"{df['identificador'].nunique():,}"
    )

    print(
        f"Date range: "
        f"{df['event_datetime'].min().date()} "
        f"-> "
        f"{df['event_datetime'].max().date()}"
    )

    return df

def read_exames_nao_regulados():
    """
    Read non-regulated examination events.
    """
    path = SOURCE_FILES[
        "exames_nao_regulados"
    ]

    print()
    print("=" * 80)
    print("4. LOADING NON-REGULATED EXAM EVENTS")
    print("=" * 80)

    usecols = [
        "datarealizacao",
        "datasolicitacao",
        "unidadesolicitante",
        "executante",
        "situacao",
        "grupoexames",
        "exame",
        "cid",
        "identificador",
    ]

    events = []

    for chunk in pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        identifier = normalize_identifier(
            chunk["identificador"]
        )

        event_datetime = pd.to_datetime(
            chunk["datarealizacao"],
            errors="coerce",
            dayfirst=True,
        )

        fallback_date = pd.to_datetime(
            chunk["datasolicitacao"],
            errors="coerce",
        )

        event_datetime = event_datetime.fillna(
            fallback_date
        )

        valid = (
            identifier.notna()
            & event_datetime.notna()
        )

        chunk = chunk.loc[
            valid
        ].copy()

        chunk["identificador"] = (
            identifier.loc[valid]
        )

        chunk["event_datetime"] = (
            event_datetime.loc[valid]
        )

        chunk["source"] = (
            "exames_nao_regulados"
        )

        chunk["is_primary_care"] = (
            chunk[
                "unidadesolicitante"
            ].apply(
                classify_primary_care_unit
            )
        )

        events.append(
            chunk[
                [
                    "identificador",
                    "event_datetime",
                    "source",
                    "is_primary_care",
                    "unidadesolicitante",
                    "executante",
                    "situacao",
                    "grupoexames",
                    "exame",
                    "cid",
                ]
            ]
        )

    if not events:
        return pd.DataFrame()

    df = pd.concat(
        events,
        ignore_index=True,
    )

    print(
        f"Non-regulated exam events: "
        f"{len(df):,}"
    )

    print(
        f"Patients: "
        f"{df['identificador'].nunique():,}"
    )

    print(
        f"Primary-care events: "
        f"{int(df['is_primary_care'].sum()):,}"
    )

    return df


def read_solicitacoes():
    """
    Read request events.
    """
    path = SOURCE_FILES[
        "solicitacoes"
    ]

    print()
    print("=" * 80)
    print("5. LOADING REQUEST EVENTS")
    print("=" * 80)

    # Read the header first because this source may have
    # slightly different column names depending on the export.
    header = pd.read_csv(
        path,
        sep=";",
        nrows=0,
        encoding="utf-8",
    ).columns.tolist()

    candidate_date_columns = [
        "dataexecucao",
        "datasolicitacao",
        "data_solicitacao",
    ]

    date_columns = [
        column
        for column in candidate_date_columns
        if column in header
    ]

    if not date_columns:
        raise ValueError(
            "No usable date column was found "
            "in solicitacoes."
        )

    usecols = [
        column
        for column in [
            "identificador",
            "datasolicitacao",
            "dataexecucao",
            "data_solicitacao",
            "unidadesolicitante",
            "unidade_solicitante",
            "executante",
            "situacao",
            "procedimentoprincipal",
            "descricao_procedimentoprincipal",
            "cidprincipal",
            "cidsecundario",
        ]
        if column in header
    ]

    events = []

    for chunk in pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        identifier = normalize_identifier(
            chunk["identificador"]
        )

        event_datetime = pd.Series(
            pd.NaT,
            index=chunk.index,
            dtype="datetime64[ns]",
        )

        for column in [
            "dataexecucao",
            "datasolicitacao",
            "data_solicitacao",
        ]:
            if column in chunk.columns:
                candidate = pd.to_datetime(
                    chunk[column],
                    errors="coerce",
                )

                event_datetime = (
                    event_datetime.fillna(
                        candidate
                    )
                )

        valid = (
            identifier.notna()
            & event_datetime.notna()
        )

        chunk = chunk.loc[
            valid
        ].copy()

        chunk["identificador"] = (
            identifier.loc[valid]
        )

        chunk["event_datetime"] = (
            event_datetime.loc[valid]
        )

        chunk["source"] = "solicitacoes"

        unit_series = None

        for column in [
            "unidadesolicitante",
            "unidade_solicitante",
        ]:
            if column in chunk.columns:
                unit_series = chunk[column]
                break

        if unit_series is None:
            chunk["is_primary_care"] = False
        else:
            chunk["is_primary_care"] = (
                unit_series.apply(
                    classify_primary_care_unit
                )
            )

        events.append(
            chunk[
                [
                    "identificador",
                    "event_datetime",
                    "source",
                    "is_primary_care",
                ]
            ]
        )

    if not events:
        return pd.DataFrame()

    df = pd.concat(
        events,
        ignore_index=True,
    )

    print(
        f"Request events: "
        f"{len(df):,}"
    )

    print(
        f"Patients: "
        f"{df['identificador'].nunique():,}"
    )

    print(
        f"Primary-care events: "
        f"{int(df['is_primary_care'].sum()):,}"
    )

    return df


def classify_primary_care_unit(value):
    """
    Heuristic classification of primary-care units.

    Used only for exploratory audit.
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
        r"ESTRATEGIA DE SAUDE DA FAMILIA",
        r"ESTRATÉGIA DE SAÚDE DA FAMÍLIA",
        r"SAUDE DA FAMILIA",
        r"SAÚDE DA FAMÍLIA",
        r"\bESF\b",
    ]

    return any(
        re.search(
            pattern,
            text,
        )
        for pattern in patterns
    )


def classify_primary_care_consultation(row):
    """
    Identify likely primary-care consultations using
    both requesting unit and CBO.
    """
    values = []

    for column in [
        "unidade_solicitante",
        "unidade_executante",
        "cbo",
    ]:
        if column in row.index:
            value = row[column]

            if pd.notna(value):
                values.append(
                    str(value)
                )

    text = " ".join(
        values
    ).upper()

    patterns = [
        r"\bUBS\b",
        r"UNIDADE DE SAUDE",
        r"UNIDADE DE SAÚDE",
        r"CLINICA DA FAMILIA",
        r"CLÍNICA DA FAMÍLIA",
        r"ESTRATEGIA DE SAUDE DA FAMILIA",
        r"ESTRATÉGIA DE SAÚDE DA FAMÍLIA",
        r"SAUDE DA FAMILIA",
        r"SAÚDE DA FAMÍLIA",
        r"\bESF\b",
    ]

    return any(
        re.search(
            pattern,
            text,
        )
        for pattern in patterns
    )


def count_events_before_admissions(
    admissions,
    events,
    source_name,
):
    """
    Count events occurring before each admission
    in 30, 90 and 365 day windows.

    Important:
        event_datetime must be strictly before admission_ts.
    """
    print()
    print("=" * 80)
    print(
        f"6. ANALYZING: {source_name}"
    )
    print("=" * 80)

    if events.empty:
        print(
            "No events available."
        )

        return pd.DataFrame()

    events = events.copy()

    events["identificador"] = (
        normalize_identifier(
            events["identificador"]
        )
    )

    events["event_datetime"] = pd.to_datetime(
        events["event_datetime"],
        errors="coerce",
    )

    events = events.dropna(
        subset=[
            "identificador",
            "event_datetime",
        ]
    )

    # Keep only patients represented in hospitalizations.
    hospital_patients = set(
        admissions["identificador"]
        .unique()
    )

    events = events[
        events["identificador"].isin(
            hospital_patients
        )
    ].copy()

    print(
        f"Events linked to hospitalization "
        f"patients: {len(events):,}"
    )

    if events.empty:
        return pd.DataFrame()

    # Sort for merge_asof.
    events["event_datetime"] = pd.to_datetime(
        events["event_datetime"],
        errors="coerce",
    ).astype("datetime64[ns]")

    admissions["admission_ts"] = pd.to_datetime(
        admissions["admission_ts"],
        errors="coerce",
    ).astype("datetime64[ns]")

    events = events.dropna(
        subset=[
            "identificador",
            "event_datetime",
        ]
    )

    admissions = admissions.dropna(
        subset=[
            "identificador",
            "admission_ts",
        ]
    )

    events = events.sort_values(
        [
            "event_datetime",
            "identificador",
        ]
    ).reset_index(drop=True)

    admissions_sorted = admissions.sort_values(
        [
            "admission_ts",
            "identificador",
        ]
    ).reset_index(drop=True)

    # For each admission, identify the most recent event
    # before the admission.
    previous = pd.merge_asof(
        admissions_sorted,
        events[
            [
                "identificador",
                "event_datetime",
            ]
        ],
        left_on="admission_ts",
        right_on="event_datetime",
        by="identificador",
        direction="backward",
        allow_exact_matches=False,
    )

    previous["days_since_event"] = (
        (
            previous["admission_ts"]
            - previous["event_datetime"]
        )
        .dt.total_seconds()
        / 86400
    )

    previous["has_event_before"] = (
        previous["event_datetime"].notna()
    )

    previous["event_last_30d"] = (
        previous["days_since_event"].between(
            0,
            30,
            inclusive="both",
        )
    )

    previous["event_last_90d"] = (
        previous["days_since_event"].between(
            0,
            90,
            inclusive="both",
        )
    )

    previous["event_last_365d"] = (
        previous["days_since_event"].between(
            0,
            365,
            inclusive="both",
        )
    )

    # Count all events in each window.
    # We use a merge based on patient and then filter
    # by temporal distance. This is done in chunks to
    # reduce memory pressure.
    result_parts = []

    admissions_small = admissions[
        [
            "episode_id",
            "identificador",
            "admission_ts",
        ]
    ].copy()

    events_small = events[
        [
            "identificador",
            "event_datetime",
            "is_primary_care",
        ]
    ].copy()

    # Group by patient to avoid a complete Cartesian product.
    events_by_patient = {
        patient: group
        for patient, group in events_small.groupby(
            "identificador"
        )
    }

    admission_groups = admissions_small.groupby(
        "identificador"
    )

    for patient, admission_group in admission_groups:
        patient_events = events_by_patient.get(
            patient
        )

        if patient_events is None:
            temp = admission_group[
                [
                    "episode_id",
                    "identificador",
                    "admission_ts",
                ]
            ].copy()

            temp["events_last_30d"] = 0
            temp["events_last_90d"] = 0
            temp["events_last_365d"] = 0
            temp["aps_events_last_30d"] = 0
            temp["aps_events_last_90d"] = 0
            temp["aps_events_last_365d"] = 0
            temp["days_since_last_event"] = pd.NA

            result_parts.append(temp)
            continue

        event_dates = (
            patient_events[
                "event_datetime"
            ]
            .values
        )

        event_aps = (
            patient_events[
                "is_primary_care"
            ]
            .fillna(False)
            .astype(bool)
            .values
        )

        rows = []

        for _, admission in admission_group.iterrows():
            admission_time = (
                admission["admission_ts"]
            )

            days = (
                (
                    admission_time
                    - patient_events[
                        "event_datetime"
                    ]
                )
                .dt.total_seconds()
                / 86400
            )

            valid = (
                days > 0
            )

            window_30 = (
                valid
                & (days <= 30)
            )

            window_90 = (
                valid
                & (days <= 90)
            )

            window_365 = (
                valid
                & (days <= 365)
            )

            rows.append(
                {
                    "episode_id": admission[
                        "episode_id"
                    ],
                    "identificador": patient,
                    "admission_ts": admission_time,
                    "events_last_30d": int(
                        window_30.sum()
                    ),
                    "events_last_90d": int(
                        window_90.sum()
                    ),
                    "events_last_365d": int(
                        window_365.sum()
                    ),
                    "aps_events_last_30d": int(
                        (
                            window_30
                            & event_aps
                        ).sum()
                    ),
                    "aps_events_last_90d": int(
                        (
                            window_90
                            & event_aps
                        ).sum()
                    ),
                    "aps_events_last_365d": int(
                        (
                            window_365
                            & event_aps
                        ).sum()
                    ),
                    "days_since_last_event": (
                        float(
                            days[valid].min()
                        )
                        if valid.any()
                        else pd.NA
                    ),
                }
            )

        result_parts.append(
            pd.DataFrame(rows)
        )

    result = pd.concat(
        result_parts,
        ignore_index=True,
    )

    print()
    print(
        f"Episodes analyzed: "
        f"{len(result):,}"
    )

    for column in [
        "events_last_30d",
        "events_last_90d",
        "events_last_365d",
        "aps_events_last_30d",
        "aps_events_last_90d",
        "aps_events_last_365d",
    ]:
        count = (
            result[column] > 0
        ).sum()

        percentage = (
            count
            / len(result)
            * 100
        )

        print(
            f"  {column}: "
            f"{count:,} episodes "
            f"({percentage:.2f}%)"
        )

    return result


def summarize_source(
    result,
    source_name,
):
    """
    Create a compact summary for one source.
    """
    if result.empty:
        return {
            "source": source_name,
            "episodes": 0,
            "episodes_with_event_30d": 0,
            "episodes_with_event_90d": 0,
            "episodes_with_event_365d": 0,
            "episodes_with_aps_30d": 0,
            "episodes_with_aps_90d": 0,
            "episodes_with_aps_365d": 0,
            "pct_event_30d": 0,
            "pct_event_90d": 0,
            "pct_event_365d": 0,
            "pct_aps_30d": 0,
            "pct_aps_90d": 0,
            "pct_aps_365d": 0,
        }

    n = len(result)

    return {
        "source": source_name,
        "episodes": n,
        "episodes_with_event_30d": int(
            (
                result[
                    "events_last_30d"
                ] > 0
            ).sum()
        ),
        "episodes_with_event_90d": int(
            (
                result[
                    "events_last_90d"
                ] > 0
            ).sum()
        ),
        "episodes_with_event_365d": int(
            (
                result[
                    "events_last_365d"
                ] > 0
            ).sum()
        ),
        "episodes_with_aps_30d": int(
            (
                result[
                    "aps_events_last_30d"
                ] > 0
            ).sum()
        ),
        "episodes_with_aps_90d": int(
            (
                result[
                    "aps_events_last_90d"
                ] > 0
            ).sum()
        ),
        "episodes_with_aps_365d": int(
            (
                result[
                    "aps_events_last_365d"
                ] > 0
            ).sum()
        ),
        "pct_event_30d": (
            result[
                "events_last_30d"
            ].gt(0).mean()
            * 100
        ),
        "pct_event_90d": (
            result[
                "events_last_90d"
            ].gt(0).mean()
            * 100
        ),
        "pct_event_365d": (
            result[
                "events_last_365d"
            ].gt(0).mean()
            * 100
        ),
        "pct_aps_30d": (
            result[
                "aps_events_last_30d"
            ].gt(0).mean()
            * 100
        ),
        "pct_aps_90d": (
            result[
                "aps_events_last_90d"
            ].gt(0).mean()
            * 100
        ),
        "pct_aps_365d": (
            result[
                "aps_events_last_365d"
            ].gt(0).mean()
            * 100
        ),
    }


def main():
    print()
    print("=" * 80)
    print("PRE-ADMISSION PATIENT TRAJECTORY AUDIT")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # 1. Hospitalization episodes
    # --------------------------------------------------------------
    admissions = (
        load_hospitalization_episodes()
    )

    # --------------------------------------------------------------
    # 2. Consultation events
    # --------------------------------------------------------------
    consultas = read_consultas()

    # --------------------------------------------------------------
    # 3. Regulated examinations
    # --------------------------------------------------------------
    exames_regulados = (
        read_exames_regulados()
    )

    # --------------------------------------------------------------
    # 4. Non-regulated examinations
    # --------------------------------------------------------------
    exames_nao_regulados = (
        read_exames_nao_regulados()
    )

    # --------------------------------------------------------------
    # 5. Requests
    # --------------------------------------------------------------
    solicitacoes = (
        read_solicitacoes()
    )

    # --------------------------------------------------------------
    # 6. Temporal analysis
    # --------------------------------------------------------------
    results = {}

    if not consultas.empty:
        results["consultas"] = (
            count_events_before_admissions(
                admissions,
                consultas,
                "consultas",
            )
        )

    if not exames_regulados.empty:
        results["exames_regulados"] = (
            count_events_before_admissions(
                admissions,
                exames_regulados,
                "exames_regulados",
            )
        )

    if not exames_nao_regulados.empty:
        results["exames_nao_regulados"] = (
            count_events_before_admissions(
                admissions,
                exames_nao_regulados,
                "exames_nao_regulados",
            )
        )

    if not solicitacoes.empty:
        results["solicitacoes"] = (
            count_events_before_admissions(
                admissions,
                solicitacoes,
                "solicitacoes",
            )
        )

    # --------------------------------------------------------------
    # 7. Summary
    # --------------------------------------------------------------
    summaries = []

    for source_name, result in results.items():
        summaries.append(
            summarize_source(
                result,
                source_name,
            )
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        OUTPUT_DIR
        / "04_pre_admission_trajectory_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------------
    # 8. Save episode-level results
    # --------------------------------------------------------------
    for source_name, result in results.items():
        output_path = (
            OUTPUT_DIR
            / f"05_{source_name}_episode_trajectory.csv"
        )

        result.to_csv(
            output_path,
            index=False,
        )

    # --------------------------------------------------------------
    # 9. Combined trajectory
    # --------------------------------------------------------------
    combined = admissions[
        [
            "episode_id",
            "identificador",
            "admission_ts",
        ]
    ].copy()

    for source_name, result in results.items():
        columns = [
            "episode_id",
            "events_last_30d",
            "events_last_90d",
            "events_last_365d",
            "aps_events_last_30d",
            "aps_events_last_90d",
            "aps_events_last_365d",
            "days_since_last_event",
        ]

        source_result = result[
            columns
        ].copy()

        source_result = source_result.rename(
            columns={
                "events_last_30d": (
                    f"{source_name}_last_30d"
                ),
                "events_last_90d": (
                    f"{source_name}_last_90d"
                ),
                "events_last_365d": (
                    f"{source_name}_last_365d"
                ),
                "aps_events_last_30d": (
                    f"{source_name}_aps_last_30d"
                ),
                "aps_events_last_90d": (
                    f"{source_name}_aps_last_90d"
                ),
                "aps_events_last_365d": (
                    f"{source_name}_aps_last_365d"
                ),
                "days_since_last_event": (
                    f"days_since_last_{source_name}"
                ),
            }
        )

        combined = combined.merge(
            source_result,
            on="episode_id",
            how="left",
        )

    numeric_columns = [
        column
        for column in combined.columns
        if column != "episode_id"
        and column != "identificador"
        and column != "admission_ts"
    ]

    combined[numeric_columns] = (
        combined[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    combined_path = (
        OUTPUT_DIR
        / "06_combined_pre_admission_trajectory.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    # --------------------------------------------------------------
    # 10. Print final summary
    # --------------------------------------------------------------
    print()
    print("=" * 80)
    print("AUDIT COMPLETED")
    print("=" * 80)

    print()
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print("Generated files:")
    print(
        f"  {summary_path}"
    )
    print(
        f"  {combined_path}"
    )

    print()
    print(
        "No ml_features table was modified."
    )

    print(
        "No model was trained."
    )

    print(
        "No test-set performance was evaluated."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()