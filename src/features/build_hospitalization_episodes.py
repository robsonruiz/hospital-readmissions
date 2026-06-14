import duckdb

def build_hospitalization_episodes():

    conn = duckdb.connect("data/hospital_readmissions.duckdb")

    conn.execute("""
        CREATE OR REPLACE TABLE hospitalization_episodes AS

        SELECT
            identificador,
            datahorainternacao,
            MAX(datahoraalta) AS datahoraalta,
            MAX(datasolicitacao) AS last_update,

            ANY_VALUE(sexo) AS sexo,
            ANY_VALUE(especialidade) AS especialidade,
            ANY_VALUE(tipoleito) AS tipoleito,
            ANY_VALUE(carater) AS carater,
            ANY_VALUE(codigocid) AS codigocid,
            ANY_VALUE(motivoalta) AS motivoalta

        FROM hospitalizations

        WHERE datahorainternacao <> 'SEM DATAHORA INTERNAÇÃO'

        GROUP BY
            identificador,
            datahorainternacao
    """)

    conn.close()


if __name__ == "__main__":
    build_hospitalization_episodes()