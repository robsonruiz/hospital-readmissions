import duckdb


def build_readmission_target():

    conn = duckdb.connect(
        "data/hospital_readmissions.duckdb"
    )

    print("Creating clean episode table...")

    conn.execute("""
        CREATE OR REPLACE TABLE hospitalization_episodes_clean AS

        SELECT
            *,

            CASE
                WHEN identificador = (
                    '504ac34ff10ba0c69bd09d503dfe942d'
                    '7576b8a843f0152b9b5e67d92bcffa65'
                )
                AND datahorainternacao = '12/11/2022 07:08:00'
                THEN STRPTIME(
                    '12/11/2025 07:08:00',
                    '%d/%m/%Y %H:%M:%S'
                )

                WHEN YEAR(
                    STRPTIME(
                        datahorainternacao,
                        '%d/%m/%Y %H:%M:%S'
                    )
                ) = 2004
                THEN STRPTIME(
                    REPLACE(datahorainternacao, '2004', '2024'),
                    '%d/%m/%Y %H:%M:%S'
                )

                WHEN YEAR(
                    STRPTIME(
                        datahorainternacao,
                        '%d/%m/%Y %H:%M:%S'
                    )
                ) = 2013
                THEN STRPTIME(
                    REPLACE(datahorainternacao, '2013', '2023'),
                    '%d/%m/%Y %H:%M:%S'
                )

                ELSE STRPTIME(
                    datahorainternacao,
                    '%d/%m/%Y %H:%M:%S'
                )
            END AS admission_ts,

            CASE
                WHEN datahoraalta IS NULL
                    OR datahoraalta = 'SEM DATAHORAALTA'
                THEN NULL

                ELSE STRPTIME(
                    datahoraalta,
                    '%d/%m/%Y %H:%M:%S'
                )
            END AS discharge_ts

        FROM hospitalization_episodes
    """)

    print("Creating readmission target...")

    conn.execute("""
        CREATE OR REPLACE TABLE hospitalization_target AS

        WITH current_episodes AS (

            SELECT
                ROW_NUMBER() OVER (
                    ORDER BY identificador, admission_ts, discharge_ts
                ) AS episode_id,

                *

            FROM hospitalization_episodes_clean
        ),

        future_candidates AS (

            SELECT
                current_episode.episode_id,

                future_episode.admission_ts AS candidate_admission_ts,
                future_episode.executante AS candidate_executante,
                future_episode.municipioexecutante
                    AS candidate_municipioexecutante

            FROM current_episodes AS current_episode

            LEFT JOIN hospitalization_episodes_clean AS future_episode

                ON future_episode.identificador =
                    current_episode.identificador

                AND current_episode.discharge_ts IS NOT NULL

                AND future_episode.admission_ts >
                    current_episode.discharge_ts
        ),

        ranked_candidates AS (

            SELECT
                *,

                ROW_NUMBER() OVER (
                    PARTITION BY episode_id
                    ORDER BY candidate_admission_ts NULLS LAST
                ) AS candidate_order

            FROM future_candidates
        ),

        episodes_with_next_admission AS (

            SELECT
                current_episode.* EXCLUDE (episode_id),

                candidate.candidate_admission_ts
                    AS next_admission_ts,

                candidate.candidate_executante
                    AS next_executante,

                candidate.candidate_municipioexecutante
                    AS next_municipioexecutante

            FROM current_episodes AS current_episode

            LEFT JOIN ranked_candidates AS candidate

                ON current_episode.episode_id =
                    candidate.episode_id

                AND candidate.candidate_order = 1
        ),

        target_logic AS (

            SELECT
                *,

                DATEDIFF(
                    'day',
                    discharge_ts,
                    next_admission_ts
                ) AS days_until_next_admission,

                CASE
                    WHEN motivoalta = 'Transferência'
                    THEN 1
                    ELSE 0
                END AS is_transfer_discharge,

                CASE
                    WHEN motivoalta = 'Óbito'
                    THEN 1
                    ELSE 0
                END AS is_death_discharge

            FROM episodes_with_next_admission
        )

        SELECT
            *,

            CASE
                WHEN next_admission_ts IS NULL
                    THEN 0

                WHEN discharge_ts IS NULL
                    THEN 0

                WHEN is_death_discharge = 1
                    THEN 0

                WHEN is_transfer_discharge = 1
                    THEN 0

                WHEN days_until_next_admission BETWEEN 0 AND 30
                    THEN 1

                ELSE 0
            END AS readmitted_30d_raw,

            CASE
                WHEN next_admission_ts IS NULL
                    THEN 0

                WHEN discharge_ts IS NULL
                    THEN 0

                WHEN is_death_discharge = 1
                    THEN 0

                WHEN is_transfer_discharge = 1
                    THEN 0

                WHEN days_until_next_admission BETWEEN 2 AND 30
                    THEN 1

                ELSE 0
            END AS readmitted_30d_clean

        FROM target_logic
    """)
    
    clean_count = conn.execute("""
        SELECT COUNT(*)
        FROM hospitalization_episodes_clean
    """).fetchone()[0]

    target_count = conn.execute("""
        SELECT COUNT(*)
        FROM hospitalization_target
    """).fetchone()[0]

    print(f"\nClean episodes: {clean_count:,}")
    print(f"Target episodes: {target_count:,}")

    if clean_count != target_count:
        raise ValueError(
            "Episode count changed while creating the target: "
            f"{clean_count:,} -> {target_count:,}"
        )

    summary = conn.execute("""
        SELECT
            readmitted_30d_clean,
            COUNT(*) AS records
        FROM hospitalization_target
        GROUP BY readmitted_30d_clean
        ORDER BY readmitted_30d_clean
    """).fetchdf()

    print("\nReadmission target distribution:")
    print(summary)

    comparison = conn.execute("""
        SELECT
            SUM(readmitted_30d_raw) AS raw_readmissions,
            SUM(readmitted_30d_clean) AS clean_readmissions
        FROM hospitalization_target
    """).fetchdf()

    print("\nRaw vs Clean target:")
    print(comparison)

    excluded = conn.execute("""
        SELECT
            motivoalta,
            COUNT(*) AS episodes
        FROM hospitalization_target
        WHERE motivoalta IN (
            'Transferência',
            'Óbito'
        )
        GROUP BY motivoalta
        ORDER BY episodes DESC
    """).fetchdf()

    print("\nExcluded discharge reasons:")
    print(excluded)

    total = conn.execute("""
        SELECT COUNT(*)
        FROM hospitalization_target
    """).fetchone()[0]

    positives = conn.execute("""
        SELECT COUNT(*)
        FROM hospitalization_target
        WHERE readmitted_30d_clean = 1
    """).fetchone()[0]

    print(f"\nTotal episodes: {total:,}")
    print(f"Readmissions within 30 days clean: {positives:,}")
    print(f"Rate: {(positives / total) * 100:.2f}%")

    conn.close()

    print("\nTarget creation completed.")


if __name__ == "__main__":
    build_readmission_target()