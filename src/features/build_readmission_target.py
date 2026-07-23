import duckdb


DATABASE_PATH = "data/hospital_readmissions.duckdb"


def build_readmission_target() -> None:
    conn = duckdb.connect(DATABASE_PATH)

    try:
        print("Creating clean episode table...")

        conn.execute("""
            CREATE OR REPLACE TABLE hospitalization_episodes_clean AS

            WITH parsed AS (
                SELECT
                    *,
                    TRY_STRPTIME(
                        datahorainternacao,
                        '%d/%m/%Y %H:%M:%S'
                    ) AS parsed_admission_ts
                FROM hospitalization_episodes
            )

            SELECT
                * EXCLUDE (parsed_admission_ts),

                CASE
                    WHEN identificador =
                        '504ac34ff10ba0c69bd09d503dfe942d7576b8a843f0152b9b5e67d92bcffa65'
                        AND datahorainternacao = '12/11/2022 07:08:00'
                    THEN TIMESTAMP '2025-11-12 07:08:00'

                    WHEN YEAR(parsed_admission_ts) = 2004
                    THEN parsed_admission_ts + INTERVAL 20 YEAR

                    WHEN YEAR(parsed_admission_ts) = 2013
                    THEN parsed_admission_ts + INTERVAL 10 YEAR

                    ELSE parsed_admission_ts
                END AS admission_ts,

                TRY_STRPTIME(
                    NULLIF(
                        NULLIF(TRIM(datahoraalta), ''),
                        'SEM DATAHORAALTA'
                    ),
                    '%d/%m/%Y %H:%M:%S'
                ) AS discharge_ts

            FROM parsed
        """)

        print("Creating unplanned-readmission target...")

        conn.execute("""
            CREATE OR REPLACE TABLE hospitalization_target AS

            WITH episodes_with_next AS (
                SELECT
                    current_episode.*,

                    next_episode.admission_ts
                        AS next_admission_ts,

                    next_episode.executante
                        AS next_executante,

                    next_episode.municipioexecutante
                        AS next_municipioexecutante,

                    next_episode.carater
                        AS next_carater,

                    next_episode.codigocid
                        AS next_codigocid,

                    DATEDIFF(
                        'day',
                        current_episode.discharge_ts,
                        next_episode.admission_ts
                    ) AS days_until_next_admission,

                    CASE
                        WHEN current_episode.motivoalta IN (
                            'Óbito',
                            'Transferência'
                        )
                            OR current_episode.discharge_ts IS NULL
                        THEN 0
                        ELSE 1
                    END AS eligible_for_readmission_model

                FROM hospitalization_episodes_clean
                    AS current_episode

                LEFT JOIN LATERAL (
                    SELECT
                        future_episode.admission_ts,
                        future_episode.executante,
                        future_episode.municipioexecutante,
                        future_episode.carater,
                        future_episode.codigocid

                    FROM hospitalization_episodes_clean
                        AS future_episode

                    WHERE future_episode.identificador =
                            current_episode.identificador

                        AND current_episode.discharge_ts IS NOT NULL

                        AND future_episode.admission_ts >
                            current_episode.discharge_ts

                    ORDER BY future_episode.admission_ts
                    LIMIT 1
                ) AS next_episode
                    ON TRUE
            ),

            raw_procedures AS (
                SELECT
                    episode.identificador,
                    episode.admission_ts,

                    STRING_AGG(
                        DISTINCT NULLIF(
                            TRIM(
                                CAST(
                                    hospitalization.procedimentosolicitado
                                    AS VARCHAR
                                )
                            ),
                            ''
                        ),
                        ' | '
                    ) AS procedure_text

                FROM hospitalizations AS hospitalization

                INNER JOIN hospitalization_episodes_clean
                    AS episode

                    ON hospitalization.identificador =
                        episode.identificador

                    AND hospitalization.datahorainternacao =
                        episode.datahorainternacao

                GROUP BY
                    episode.identificador,
                    episode.admission_ts
            ),

            billing_procedures AS (
                SELECT
                    identificador,
                    datahorainternacao AS admission_ts,

                    STRING_AGG(
                        DISTINCT NULLIF(
                            TRIM(
                                CONCAT_WS(
                                    ' | ',
                                    CAST(
                                        procedimentosolicitado
                                        AS VARCHAR
                                    ),
                                    CAST(
                                        procprincipalsolicitado
                                        AS VARCHAR
                                    ),
                                    CAST(
                                        procprincipalautorizado
                                        AS VARCHAR
                                    ),
                                    CAST(
                                        procedimentosolicitadoeletiva
                                        AS VARCHAR
                                    )
                                )
                            ),
                            ''
                        ),
                        ' | '
                    ) AS procedure_text

                FROM billing

                GROUP BY
                    identificador,
                    datahorainternacao
            ),

            candidates AS (
                SELECT
                    episode.identificador,
                    episode.admission_ts,

                    UPPER(
                        COALESCE(
                            episode.next_carater,
                            ''
                        )
                    ) AS admission_character,

                    UPPER(
                        REPLACE(
                            REPLACE(
                                COALESCE(
                                    episode.next_codigocid,
                                    ''
                                ),
                                '.',
                                ''
                            ),
                            '-',
                            ''
                        )
                    ) AS cid,

                    UPPER(
                        CONCAT_WS(
                            ' | ',
                            raw.procedure_text,
                            billing.procedure_text
                        )
                    ) AS procedure_text

                FROM episodes_with_next AS episode

                LEFT JOIN raw_procedures AS raw
                    ON episode.identificador =
                        raw.identificador

                    AND episode.next_admission_ts =
                        raw.admission_ts

                LEFT JOIN billing_procedures AS billing
                    ON episode.identificador =
                        billing.identificador

                    AND episode.next_admission_ts =
                        billing.admission_ts

                WHERE episode.eligible_for_readmission_model = 1
                    AND episode.days_until_next_admission
                        BETWEEN 2 AND 30
            ),

            signals AS (
                SELECT
                    *,

                    CASE
                        WHEN REGEXP_MATCHES(
                            cid,
                            '^(A0|A39|A40|A41|A49|I21|I22|I6[0-4]|J1[2-8]|J80|J96|K65|N10|N12|N17|N30|N39|R57|T8[0-8]|P95)'
                        )
                        OR (
                            cid LIKE 'O%'
                            AND (
                                TRY_CAST(
                                    SUBSTR(cid, 2, 2)
                                    AS INTEGER
                                ) BETWEEN 0 AND 8

                                OR TRY_CAST(
                                    SUBSTR(cid, 2, 2)
                                    AS INTEGER
                                ) BETWEEN 10 AND 29

                                OR TRY_CAST(
                                    SUBSTR(cid, 2, 2)
                                    AS INTEGER
                                ) BETWEEN 42 AND 46

                                OR TRY_CAST(
                                    SUBSTR(cid, 2, 2)
                                    AS INTEGER
                                ) BETWEEN 60 AND 75

                                OR TRY_CAST(
                                    SUBSTR(cid, 2, 2)
                                    AS INTEGER
                                ) BETWEEN 85 AND 99
                            )
                        )
                        THEN 1
                        ELSE 0
                    END AS acute_cid_signal,

                    CASE
                        WHEN REGEXP_MATCHES(
                            procedure_text,
                            'INTERCORR|COMPLICA|ATENDIMENTO DE URGENC|SINDROME CORONARIANA AGUDA|INSUFICIENCIA RENAL AGUDA|CRISES EPILEPTICAS|PNEUMON|PIELONEFR|SEPSE|ACIDENTE VASCULAR CEREBRAL|INFART|HEMORR|INFECC'
                        )
                        THEN 1
                        ELSE 0
                    END AS acute_procedure_signal

                FROM candidates
            ),

            classified AS (
                SELECT
                    identificador,
                    admission_ts,

                    CASE
                        WHEN admission_character
                            LIKE '%TRANSFER%'
                        THEN 0

                        WHEN acute_cid_signal = 1
                            OR acute_procedure_signal = 1
                        THEN 1

                        WHEN (
                            admission_character LIKE '%URGEN%'
                            OR admission_character
                                LIKE '%EMERGEN%'
                        )
                        AND NOT (
                            cid LIKE 'O80%'
                            OR cid LIKE 'O82%'

                            OR REGEXP_MATCHES(
                                procedure_text,
                                'PARTO|CESARI|QUIMIOTERAP|RADIOTERAP|ANTINEOPLAS|HEMODIAL|DI[AÁ]LISE|REABILIT'
                            )

                            OR (
                                procedure_text
                                    LIKE '%TRANSPLANTE DE%'

                                AND procedure_text
                                    NOT LIKE '%POS-TRANSPLANTE%'

                                AND procedure_text
                                    NOT LIKE '%PÓS-TRANSPLANTE%'
                            )
                        )
                        THEN 1

                        ELSE 0
                    END AS unplanned_readmitted_30d

                FROM signals
            )

            SELECT
                episode.* EXCLUDE (
                    next_carater,
                    next_codigocid
                ),

                CASE
                    WHEN episode.eligible_for_readmission_model = 0
                    THEN NULL

                    ELSE CAST(
                        COALESCE(
                            classified.unplanned_readmitted_30d,
                            0
                        )
                        AS INTEGER
                    )
                END AS unplanned_readmitted_30d

            FROM episodes_with_next AS episode

            LEFT JOIN classified
                ON episode.identificador =
                    classified.identificador

                AND episode.admission_ts =
                    classified.admission_ts
        """)

        clean_count = conn.execute("""
            SELECT COUNT(*)
            FROM hospitalization_episodes_clean
        """).fetchone()[0]

        target_count = conn.execute("""
            SELECT COUNT(*)
            FROM hospitalization_target
        """).fetchone()[0]

        if clean_count != target_count:
            raise ValueError(
                "Episode count changed: "
                f"{clean_count:,} -> {target_count:,}"
            )

        invalid_count = conn.execute("""
            SELECT COUNT(*)
            FROM hospitalization_target
            WHERE
                (
                    eligible_for_readmission_model = 0
                    AND unplanned_readmitted_30d
                        IS NOT NULL
                )
                OR (
                    eligible_for_readmission_model = 1
                    AND unplanned_readmitted_30d
                        IS NULL
                )
                OR (
                    unplanned_readmitted_30d = 1
                    AND days_until_next_admission
                        NOT BETWEEN 2 AND 30
                )
        """).fetchone()[0]

        if invalid_count:
            raise ValueError(
                f"Invalid target records: {invalid_count:,}"
            )

        summary = conn.execute("""
            SELECT
                COUNT(*) AS total_episodes,

                COUNT(*) FILTER (
                    WHERE eligible_for_readmission_model = 1
                ) AS eligible_episodes,

                COUNT(*) FILTER (
                    WHERE unplanned_readmitted_30d = 1
                ) AS unplanned_readmissions,

                ROUND(
                    COUNT(*) FILTER (
                        WHERE unplanned_readmitted_30d = 1
                    ) * 100.0
                    / NULLIF(
                        COUNT(*) FILTER (
                            WHERE
                                eligible_for_readmission_model = 1
                        ),
                        0
                    ),
                    2
                ) AS unplanned_readmission_rate

            FROM hospitalization_target
        """).fetchdf()

        print(summary)
        print("Target creation completed.")

    finally:
        conn.close()


if __name__ == "__main__":
    build_readmission_target()