import duckdb

df = duckdb.sql("""
SELECT *
FROM 'data/raw/1_consultas_gercon.csv'
LIMIT 1000
""").df()