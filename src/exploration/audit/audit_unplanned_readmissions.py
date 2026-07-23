from pathlib import Path

import duckdb
import pandas as pd


DB_PATH = Path("data/hospital_readmissions.duckdb")

OUTPUT_DIR = Path(
    "data/quality_audit/unplanned_readmissions"
)

TARGET_TABLE = "hospitalization_target"
EPISODES_TABLE = "hospitalization_episodes_clean"


def get_columns(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> set[str]:
    """
    Retorna as colunas disponíveis em uma tabela.
    """

    result = connection.execute(
        f"PRAGMA table_info('{table_name}')"
    ).fetchall()

    return {row[1] for row in result}


def print_and_save(
    connection: duckdb.DuckDBPyConnection,
    title: str,
    filename: str,
    query: str,
) -> pd.DataFrame:
    """
    Executa uma consulta, exibe o resultado e salva em CSV.
    """

    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)

    result = connection.execute(query).df()

    if result.empty:
        print("Nenhum resultado encontrado.")
    else:
        print(result.to_string(index=False))

    output_path = OUTPUT_DIR / filename

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nArquivo salvo em: {output_path}")

    return result


def create_temporary_readmission_view(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Cria uma visão temporária ligando o episódio índice
    à internação subsequente.

    Nenhuma coluna é criada permanentemente no banco.
    """

    target_columns = get_columns(
        connection,
        TARGET_TABLE,
    )

    episode_columns = get_columns(
        connection,
        EPISODES_TABLE,
    )

    required_target_columns = {
        "identificador",
        "admission_ts",
        "next_admission_ts",
        "readmitted_30d_clean",
    }

    required_episode_columns = {
        "identificador",
        "admission_ts",
        "carater",
        "especialidade",
        "codigocid",
    }

    missing_target = (
        required_target_columns - target_columns
    )

    missing_episodes = (
        required_episode_columns - episode_columns
    )

    if missing_target:
        raise ValueError(
            "Colunas ausentes em "
            f"{TARGET_TABLE}: "
            f"{sorted(missing_target)}"
        )

    if missing_episodes:
        raise ValueError(
            "Colunas ausentes em "
            f"{EPISODES_TABLE}: "
            f"{sorted(missing_episodes)}"
        )

    select_columns = [
        "target.identificador",
        (
            "target.admission_ts "
            "AS index_admission_ts"
        ),
        "target.next_admission_ts",
        "target.readmitted_30d_clean",
    ]

    optional_target_columns = {
        "discharge_ts": "index_discharge_ts",
        "days_until_next_admission": (
            "days_until_next_admission"
        ),
        "eligible_for_readmission_model": (
            "eligible_for_readmission_model"
        ),
        "carater": "index_carater",
        "especialidade": "index_especialidade",
        "codigocid": "index_codigocid",
        "motivoalta": "index_motivoalta",
        "tipoleito": "index_tipoleito",
    }

    for column, alias in optional_target_columns.items():
        if column in target_columns:
            select_columns.append(
                f"target.{column} AS {alias}"
            )

    next_episode_columns = {
        "admission_ts": "matched_next_admission_ts",
        "carater": "next_carater",
        "especialidade": "next_especialidade",
        "codigocid": "next_codigocid",
        "motivoalta": "next_motivoalta",
        "tipoleito": "next_tipoleito",
        "executante": "next_executante",
        "municipioexecutante": (
            "next_municipioexecutante"
        ),
    }

    for column, alias in next_episode_columns.items():
        if column in episode_columns:
            select_columns.append(
                f"next_episode.{column} AS {alias}"
            )

    select_sql = ",\n            ".join(
        select_columns
    )

    query = f"""
        CREATE OR REPLACE TEMP VIEW
            readmissions_enriched
        AS
        SELECT
            {select_sql}

        FROM {TARGET_TABLE} AS target

        LEFT JOIN {EPISODES_TABLE}
            AS next_episode

            ON target.identificador
                = next_episode.identificador

            AND target.next_admission_ts
                = next_episode.admission_ts

        WHERE target.readmitted_30d_clean = 1
    """

    connection.execute(query)

    print(
        "\nVisão temporária "
        "'readmissions_enriched' criada."
    )


def audit_join_coverage(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Verifica se a internação subsequente foi localizada.
    """

    print_and_save(
        connection,
        "COBERTURA DO JOIN COM A INTERNAÇÃO SUBSEQUENTE",
        "01_join_coverage.csv",
        """
        SELECT
            COUNT(*) AS total_readmissions,

            COUNT(*) FILTER (
                WHERE matched_next_admission_ts
                    IS NOT NULL
            ) AS matched_readmissions,

            COUNT(*) FILTER (
                WHERE matched_next_admission_ts
                    IS NULL
            ) AS unmatched_readmissions,

            ROUND(
                COUNT(*) FILTER (
                    WHERE matched_next_admission_ts
                        IS NOT NULL
                )
                * 100.0
                / NULLIF(COUNT(*), 0),
                2
            ) AS match_percentage

        FROM readmissions_enriched
        """,
    )


def audit_next_admission_character(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Analisa o caráter da internação subsequente.
    """

    print_and_save(
        connection,
        "CARÁTER DA INTERNAÇÃO SUBSEQUENTE",
        "02_next_admission_character.csv",
        """
        SELECT
            COALESCE(
                next_carater,
                'Não informado'
            ) AS next_carater,

            COUNT(*) AS readmissions,

            ROUND(
                COUNT(*) * 100.0
                / SUM(COUNT(*)) OVER (),
                2
            ) AS percentage

        FROM readmissions_enriched

        GROUP BY 1
        ORDER BY readmissions DESC
        """,
    )


def audit_planning_proxy(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Cria uma classificação exploratória usando somente
    o caráter da internação subsequente.

    Eletivo/programado:
        potencialmente planejada.

    Urgência/emergência:
        potencialmente não planejada.

    Outros ou ausentes:
        indeterminada.
    """

    print_and_save(
        connection,
        "CLASSIFICAÇÃO EXPLORATÓRIA PELO CARÁTER",
        "03_planning_proxy.csv",
        """
        WITH classified AS (
            SELECT
                CASE
                    WHEN next_carater IS NULL
                        OR TRIM(next_carater) = ''
                        THEN 'Indeterminada'

                    WHEN
                        UPPER(next_carater)
                            LIKE '%ELETIV%'
                        OR UPPER(next_carater)
                            LIKE '%PROGRAM%'
                        THEN
                            'Potencialmente planejada'

                    WHEN
                        UPPER(next_carater)
                            LIKE '%URGEN%'
                        OR UPPER(next_carater)
                            LIKE '%EMERGEN%'
                        THEN
                            'Potencialmente não planejada'

                    ELSE 'Indeterminada'
                END AS planning_classification

            FROM readmissions_enriched
        )

        SELECT
            planning_classification,
            COUNT(*) AS readmissions,

            ROUND(
                COUNT(*) * 100.0
                / SUM(COUNT(*)) OVER (),
                2
            ) AS percentage

        FROM classified

        GROUP BY 1
        ORDER BY readmissions DESC
        """,
    )


def audit_specialty_by_character(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Cruza especialidade e caráter da internação subsequente.
    """

    print_and_save(
        connection,
        "ESPECIALIDADE POR CARÁTER DA REINTERNAÇÃO",
        "04_specialty_by_character.csv",
        """
        SELECT
            COALESCE(
                next_carater,
                'Não informado'
            ) AS next_carater,

            COALESCE(
                next_especialidade,
                'Não informado'
            ) AS next_especialidade,

            COUNT(*) AS readmissions,

            ROUND(
                COUNT(*) * 100.0
                / SUM(COUNT(*)) OVER (
                    PARTITION BY
                        COALESCE(
                            next_carater,
                            'Não informado'
                        )
                ),
                2
            ) AS percentage_within_character

        FROM readmissions_enriched

        GROUP BY 1, 2

        ORDER BY
            next_carater,
            readmissions DESC
        """,
    )


def audit_routine_specialties(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Analisa grupos que podem conter internações programadas.

    Esses grupos são somente categorias exploratórias.
    A especialidade não deve ser usada isoladamente
    para excluir uma reinternação.
    """

    print_and_save(
        connection,
        "ESPECIALIDADES POTENCIALMENTE ASSOCIADAS A ROTINA",
        "05_routine_specialties.csv",
        """
        WITH specialty_groups AS (
            SELECT
                CASE
                    WHEN
                        UPPER(
                            COALESCE(
                                next_especialidade,
                                ''
                            )
                        ) LIKE '%NEFRO%'
                        OR UPPER(
                            COALESCE(
                                next_especialidade,
                                ''
                            )
                        ) LIKE '%DIAL%'
                        THEN 'Nefrologia ou diálise'

                    WHEN
                        UPPER(
                            COALESCE(
                                next_especialidade,
                                ''
                            )
                        ) LIKE '%ONCO%'
                        OR UPPER(
                            COALESCE(
                                next_especialidade,
                                ''
                            )
                        ) LIKE '%HEMATO%'
                        THEN 'Oncologia ou hematologia'

                    WHEN
                        UPPER(
                            COALESCE(
                                next_especialidade,
                                ''
                            )
                        ) LIKE '%REABIL%'
                        THEN 'Reabilitação'

                    WHEN
                        UPPER(
                            COALESCE(
                                next_especialidade,
                                ''
                            )
                        ) LIKE '%OBSTET%'
                        OR UPPER(
                            COALESCE(
                                next_especialidade,
                                ''
                            )
                        ) LIKE '%GINECO%'
                        THEN 'Obstetrícia ou ginecologia'

                    WHEN
                        UPPER(
                            COALESCE(
                                next_especialidade,
                                ''
                            )
                        ) LIKE '%NEONAT%'
                        THEN 'Neonatologia'

                    WHEN
                        UPPER(
                            COALESCE(
                                next_especialidade,
                                ''
                            )
                        ) LIKE '%PSIQUI%'
                        OR UPPER(
                            COALESCE(
                                next_especialidade,
                                ''
                            )
                        ) LIKE '%MENTAL%'
                        THEN 'Saúde mental'

                    ELSE 'Outras especialidades'
                END AS specialty_group,

                COALESCE(
                    next_carater,
                    'Não informado'
                ) AS next_carater

            FROM readmissions_enriched
        )

        SELECT
            specialty_group,
            next_carater,
            COUNT(*) AS readmissions,

            ROUND(
                COUNT(*) * 100.0
                / SUM(COUNT(*)) OVER (
                    PARTITION BY specialty_group
                ),
                2
            ) AS percentage_within_group

        FROM specialty_groups

        GROUP BY 1, 2

        ORDER BY
            specialty_group,
            readmissions DESC
        """,
    )


def audit_elective_specialties(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Mostra quais especialidades aparecem nas
    reinternações eletivas.
    """

    print_and_save(
        connection,
        "ESPECIALIDADES DAS REINTERNAÇÕES ELETIVAS",
        "06_elective_specialties.csv",
        """
        SELECT
            COALESCE(
                next_especialidade,
                'Não informado'
            ) AS next_especialidade,

            COUNT(*) AS elective_readmissions,

            ROUND(
                COUNT(*) * 100.0
                / SUM(COUNT(*)) OVER (),
                2
            ) AS percentage

        FROM readmissions_enriched

        WHERE
            UPPER(
                COALESCE(next_carater, '')
            ) LIKE '%ELETIV%'

            OR UPPER(
                COALESCE(next_carater, '')
            ) LIKE '%PROGRAM%'

        GROUP BY 1
        ORDER BY elective_readmissions DESC
        """,
    )


def audit_urgent_specialties(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Mostra quais especialidades aparecem nas
    reinternações urgentes.
    """

    print_and_save(
        connection,
        "ESPECIALIDADES DAS REINTERNAÇÕES URGENTES",
        "07_urgent_specialties.csv",
        """
        SELECT
            COALESCE(
                next_especialidade,
                'Não informado'
            ) AS next_especialidade,

            COUNT(*) AS urgent_readmissions,

            ROUND(
                COUNT(*) * 100.0
                / SUM(COUNT(*)) OVER (),
                2
            ) AS percentage

        FROM readmissions_enriched

        WHERE
            UPPER(
                COALESCE(next_carater, '')
            ) LIKE '%URGEN%'

            OR UPPER(
                COALESCE(next_carater, '')
            ) LIKE '%EMERGEN%'

        GROUP BY 1
        ORDER BY urgent_readmissions DESC
        """,
    )


def audit_cid_by_character(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Lista os diagnósticos mais frequentes segundo
    o caráter da internação subsequente.
    """

    print_and_save(
        connection,
        "CID DA REINTERNAÇÃO POR CARÁTER",
        "08_cid_by_character.csv",
        """
        SELECT
            COALESCE(
                next_carater,
                'Não informado'
            ) AS next_carater,

            COALESCE(
                next_codigocid,
                'Não informado'
            ) AS next_codigocid,

            COUNT(*) AS readmissions

        FROM readmissions_enriched

        GROUP BY 1, 2

        ORDER BY
            next_carater,
            readmissions DESC
        """,
    )


def compare_readmission_rates(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Compara a taxa geral com uma taxa conservadora de
    reinternação não planejada.

    A taxa conservadora considera somente internações
    subsequentes marcadas como urgência ou emergência.
    """

    target_columns = get_columns(
        connection,
        TARGET_TABLE,
    )

    if (
        "eligible_for_readmission_model"
        in target_columns
    ):
        denominator_filter = (
            "eligible_for_readmission_model = 1"
        )

        enriched_filter = (
            "eligible_for_readmission_model = 1"
        )
    else:
        denominator_filter = "TRUE"
        enriched_filter = "TRUE"

    query = f"""
        WITH denominator AS (
            SELECT
                COUNT(*) AS eligible_episodes,

                SUM(readmitted_30d_clean)
                    AS all_readmissions

            FROM {TARGET_TABLE}

            WHERE {denominator_filter}
        ),

        classified_readmissions AS (
            SELECT
                CASE
                    WHEN
                        UPPER(
                            COALESCE(
                                next_carater,
                                ''
                            )
                        ) LIKE '%ELETIV%'

                        OR UPPER(
                            COALESCE(
                                next_carater,
                                ''
                            )
                        ) LIKE '%PROGRAM%'

                        THEN
                            'Potencialmente planejada'

                    WHEN
                        UPPER(
                            COALESCE(
                                next_carater,
                                ''
                            )
                        ) LIKE '%URGEN%'

                        OR UPPER(
                            COALESCE(
                                next_carater,
                                ''
                            )
                        ) LIKE '%EMERGEN%'

                        THEN
                            'Potencialmente não planejada'

                    ELSE 'Indeterminada'
                END AS planning_classification

            FROM readmissions_enriched

            WHERE {enriched_filter}
        )

        SELECT
            denominator.eligible_episodes,

            denominator.all_readmissions,

            COUNT(*) FILTER (
                WHERE planning_classification
                    = 'Potencialmente planejada'
            ) AS potentially_planned_readmissions,

            COUNT(*) FILTER (
                WHERE planning_classification
                    = 'Potencialmente não planejada'
            ) AS potentially_unplanned_readmissions,

            COUNT(*) FILTER (
                WHERE planning_classification
                    = 'Indeterminada'
            ) AS indeterminate_readmissions,

            ROUND(
                denominator.all_readmissions
                * 100.0
                / NULLIF(
                    denominator.eligible_episodes,
                    0
                ),
                2
            ) AS general_readmission_rate,

            ROUND(
                COUNT(*) FILTER (
                    WHERE planning_classification
                        = 'Potencialmente não planejada'
                )
                * 100.0
                / NULLIF(
                    denominator.eligible_episodes,
                    0
                ),
                2
            ) AS conservative_unplanned_rate

        FROM classified_readmissions

        CROSS JOIN denominator

        GROUP BY
            denominator.eligible_episodes,
            denominator.all_readmissions
    """

    print_and_save(
        connection,
        "COMPARAÇÃO DAS TAXAS DE REINTERNAÇÃO",
        "09_rate_comparison.csv",
        query,
    )


def export_review_sample(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """
    Exporta uma amostra para revisão manual.
    """

    print_and_save(
        connection,
        "AMOSTRA DE REINTERNAÇÕES PARA REVISÃO",
        "10_manual_review_sample.csv",
        """
        SELECT
            *,

            CASE
                WHEN
                    UPPER(
                        COALESCE(next_carater, '')
                    ) LIKE '%ELETIV%'

                    OR UPPER(
                        COALESCE(next_carater, '')
                    ) LIKE '%PROGRAM%'

                    THEN
                        'Potencialmente planejada'

                WHEN
                    UPPER(
                        COALESCE(next_carater, '')
                    ) LIKE '%URGEN%'

                    OR UPPER(
                        COALESCE(next_carater, '')
                    ) LIKE '%EMERGEN%'

                    THEN
                        'Potencialmente não planejada'

                ELSE 'Indeterminada'
            END AS exploratory_classification

        FROM readmissions_enriched

        ORDER BY RANDOM()

        LIMIT 300
        """,
    )


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Banco DuckDB não encontrado em: "
            f"{DB_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )

    try:
        create_temporary_readmission_view(
            connection
        )

        audit_join_coverage(connection)

        audit_next_admission_character(
            connection
        )

        audit_planning_proxy(connection)

        audit_specialty_by_character(
            connection
        )

        audit_routine_specialties(
            connection
        )

        audit_elective_specialties(
            connection
        )

        audit_urgent_specialties(
            connection
        )

        audit_cid_by_character(
            connection
        )

        compare_readmission_rates(
            connection
        )

        export_review_sample(
            connection
        )

    finally:
        connection.close()

    print("\n")
    print("=" * 100)
    print("AUDITORIA FINALIZADA")
    print("=" * 100)

    print(
        f"Resultados disponíveis em: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()