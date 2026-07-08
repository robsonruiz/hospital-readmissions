import duckdb

conn = duckdb.connect("data/hospital_readmissions.duckdb")

print(conn.execute("""
SELECT
    COUNT(*) AS total_rows,

    COUNT(DISTINCT CONCAT(
        identificador, '|',
        COALESCE(datahorainternacao, '')
    )) AS by_patient_admission,

    COUNT(DISTINCT CONCAT(
        identificador, '|',
        COALESCE(CAST(datasolicitacao AS VARCHAR), '')
    )) AS by_patient_request_date,

    COUNT(DISTINCT CONCAT(
        identificador, '|',
        COALESCE(CAST(datasolicitacao AS VARCHAR), ''), '|',
        COALESCE(datahorainternacao, '')
    )) AS by_patient_request_admission

FROM hospitalizations
WHERE datahorainternacao IS NOT NULL
  AND datahorainternacao <> 'SEM DATAHORA INTERNAÇÃO'
""").fetchdf())

conn.close()