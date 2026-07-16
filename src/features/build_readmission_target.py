import duckdb


DATABASE_PATH = "data/hospital_readmissions.duckdb"


def build_readmission_target():

    conn = duckdb.connect(DATABASE_PATH)

    try:
        # ======================================================
        # 1. Clean and standardize hospitalization dates
        # ======================================================

        print("Creating clean episode table...")

        conn.execute("""
            CREATE OR REPLACE TABLE hospitalization_episodes_clean AS

            SELECT
                *,

                CASE
                    -- Confirmed correction for one specific record
                    WHEN identificador =
                        '504ac34ff10ba0c69bd09d503dfe942d7576b8a843f0152b9b5e67d92bcffa65'
                        AND datahorainternacao = '12/11/2022 07:08:00'
                    THEN STRPTIME(
                        '12/11/2025 07:08:00',
                        '%d/%m/%Y %H:%M:%S'
                    )

                    -- Confirmed correction: 2004 -> 2024
                    WHEN YEAR(
                        STRPTIME(
                            datahorainternacao,
                            '%d/%m/%Y %H:%M:%S'
                        )
                    ) = 2004
                    THEN STRPTIME(
                        REPLACE(
                            datahorainternacao,
                            '2004',
                            '2024'
                        ),
                        '%d/%m/%Y %H:%M:%S'
                    )

                    -- Confirmed correction: 2013 -> 2023
                    WHEN YEAR(
                        STRPTIME(
                            datahorainternacao,
                            '%d/%m/%Y %H:%M:%S'
                        )
                    ) = 2013
                    THEN STRPTIME(
                        REPLACE(
                            datahorainternacao,
                            '2013',
                            '2023'
                        ),
                        '%d/%m/%Y %H:%M:%S'
                    )

                    ELSE STRPTIME(
                        datahorainternacao,
                        '%d/%m/%Y %H:%M:%S'
                    )

                END AS admission_ts,

                CASE
                    WHEN datahoraalta IS NULL
                        OR TRIM(datahoraalta) = ''
                        OR datahoraalta = 'SEM DATAHORAALTA'
                    THEN NULL

                    ELSE STRPTIME(
                        datahoraalta,
                        '%d/%m/%Y %H:%M:%S'
                    )

                END AS discharge_ts

            FROM hospitalization_episodes
        """)

        # ======================================================
        # 2. Identify first valid admission after each discharge
        # ======================================================

        print("Creating readmission target...")

        conn.execute("""
            CREATE OR REPLACE TABLE hospitalization_target AS

            WITH current_episodes AS (

                SELECT
                    ROW_NUMBER() OVER (
                        ORDER BY
                            identificador,
                            admission_ts,
                            discharge_ts
                    ) AS episode_id,

                    *

                FROM hospitalization_episodes_clean
            ),

            future_candidates AS (

                SELECT
                    current_episode.episode_id,

                    future_episode.admission_ts
                        AS candidate_admission_ts,

                    future_episode.executante
                        AS candidate_executante,

                    future_episode.municipioexecutante
                        AS candidate_municipioexecutante

                FROM current_episodes AS current_episode

                LEFT JOIN hospitalization_episodes_clean
                    AS future_episode

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
                        ORDER BY
                            candidate_admission_ts NULLS LAST
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
                    END AS is_death_discharge,

                    CASE
                        WHEN discharge_ts IS NULL
                        THEN 1
                        ELSE 0
                    END AS is_missing_discharge

                FROM episodes_with_next_admission
            )

            SELECT
                *,

                -- Target including same-day and one-day returns
                CASE
                    WHEN next_admission_ts IS NULL
                    THEN 0

                    WHEN discharge_ts IS NULL
                    THEN 0

                    WHEN is_death_discharge = 1
                    THEN 0

                    WHEN is_transfer_discharge = 1
                    THEN 0

                    WHEN days_until_next_admission
                        BETWEEN 0 AND 30
                    THEN 1

                    ELSE 0

                END AS readmitted_30d_raw,

                -- Clean target: only returns from 2 to 30 days
                CASE
                    WHEN next_admission_ts IS NULL
                    THEN 0

                    WHEN discharge_ts IS NULL
                    THEN 0

                    WHEN is_death_discharge = 1
                    THEN 0

                    WHEN is_transfer_discharge = 1
                    THEN 0

                    WHEN days_until_next_admission
                        BETWEEN 2 AND 30
                    THEN 1

                    ELSE 0

                END AS readmitted_30d_clean,

                -- Selection flag for the ML population
                CASE
                    WHEN is_death_discharge = 1
                    THEN 0

                    WHEN is_transfer_discharge = 1
                    THEN 0

                    WHEN is_missing_discharge = 1
                    THEN 0

                    ELSE 1

                END AS eligible_for_readmission_model

            FROM target_logic
        """)

        # ======================================================
        # 3. Validate row preservation
        # ======================================================

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

        # ======================================================
        # 4. Target distribution
        # ======================================================

        summary = conn.execute("""
            SELECT
                readmitted_30d_clean,
                COUNT(*) AS records,

                ROUND(
                    COUNT(*) * 100.0
                    / SUM(COUNT(*)) OVER (),
                    2
                ) AS percentage

            FROM hospitalization_target

            GROUP BY readmitted_30d_clean
            ORDER BY readmitted_30d_clean
        """).fetchdf()

        print("\nReadmission target distribution — full dataset:")
        print(summary)

        # ======================================================
        # 5. Raw vs clean target
        # ======================================================

        comparison = conn.execute("""
            SELECT
                SUM(readmitted_30d_raw)
                    AS raw_readmissions,

                SUM(readmitted_30d_clean)
                    AS clean_readmissions,

                COUNT(*) AS total_episodes,

                ROUND(
                    AVG(readmitted_30d_raw) * 100,
                    2
                ) AS raw_rate,

                ROUND(
                    AVG(readmitted_30d_clean) * 100,
                    2
                ) AS clean_rate

            FROM hospitalization_target
        """).fetchdf()

        print("\nRaw vs clean target — full dataset:")
        print(comparison)

        # ======================================================
        # 6. Model eligibility
        # ======================================================

        eligibility = conn.execute("""
            SELECT
                eligible_for_readmission_model,
                COUNT(*) AS episodes,

                ROUND(
                    COUNT(*) * 100.0
                    / SUM(COUNT(*)) OVER (),
                    2
                ) AS percentage

            FROM hospitalization_target

            GROUP BY eligible_for_readmission_model
            ORDER BY eligible_for_readmission_model
        """).fetchdf()

        print("\nEligibility for readmission model:")
        print(eligibility)

        # ======================================================
        # 7. Non-eligible reasons
        # ======================================================

        non_eligible_reasons = conn.execute("""
            SELECT
                CASE
                    WHEN motivoalta = 'Óbito'
                    THEN 'Óbito'

                    WHEN motivoalta = 'Transferência'
                    THEN 'Transferência'

                    WHEN discharge_ts IS NULL
                    THEN 'Alta ausente'

                    ELSE 'Outro'
                END AS non_eligibility_reason,

                COUNT(*) AS episodes

            FROM hospitalization_target

            WHERE eligible_for_readmission_model = 0

            GROUP BY non_eligibility_reason
            ORDER BY episodes DESC
        """).fetchdf()

        print("\nNon-eligible episodes by reason:")
        print(non_eligible_reasons)

        # ======================================================
        # 8. Distribution among eligible episodes
        # ======================================================

        eligible_distribution = conn.execute("""
            SELECT
                readmitted_30d_clean,
                COUNT(*) AS records,

                ROUND(
                    COUNT(*) * 100.0
                    / SUM(COUNT(*)) OVER (),
                    2
                ) AS percentage

            FROM hospitalization_target

            WHERE eligible_for_readmission_model = 1

            GROUP BY readmitted_30d_clean
            ORDER BY readmitted_30d_clean
        """).fetchdf()

        print("\nReadmission distribution — eligible episodes only:")
        print(eligible_distribution)

        # ======================================================
        # 9. Discharge outcome summary
        # ======================================================

        discharge_summary = conn.execute("""
            SELECT
                COALESCE(motivoalta, 'Não informado')
                    AS motivoalta,

                COUNT(*) AS episodes,

                SUM(readmitted_30d_clean)
                    AS readmissions,

                MAX(eligible_for_readmission_model)
                    AS eligible_category

            FROM hospitalization_target

            GROUP BY
                COALESCE(motivoalta, 'Não informado')

            ORDER BY episodes DESC
        """).fetchdf()

        print("\nEpisodes by discharge outcome:")
        print(discharge_summary)

        # ======================================================
        # 10. Final summary
        # ======================================================

        total = target_count

        positives = conn.execute("""
            SELECT COUNT(*)
            FROM hospitalization_target
            WHERE readmitted_30d_clean = 1
        """).fetchone()[0]

        eligible_total = conn.execute("""
            SELECT COUNT(*)
            FROM hospitalization_target
            WHERE eligible_for_readmission_model = 1
        """).fetchone()[0]

        eligible_positives = conn.execute("""
            SELECT COUNT(*)
            FROM hospitalization_target
            WHERE eligible_for_readmission_model = 1
              AND readmitted_30d_clean = 1
        """).fetchone()[0]

        overall_rate = (
            positives / total * 100
            if total > 0
            else 0
        )

        eligible_rate = (
            eligible_positives / eligible_total * 100
            if eligible_total > 0
            else 0
        )

        print(f"\nTotal episodes: {total:,}")
        print(f"Readmissions in full dataset: {positives:,}")
        print(f"Full-dataset rate: {overall_rate:.2f}%")

        print(f"\nEligible episodes for ML: {eligible_total:,}")
        print(
            "Readmissions among eligible episodes: "
            f"{eligible_positives:,}"
        )
        print(
            "Eligible-population readmission rate: "
            f"{eligible_rate:.2f}%"
        )

        print("\nTarget creation completed.")

    finally:
        conn.close()


if __name__ == "__main__":
    build_readmission_target()