import gzip
import os
import time
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = next(Path("data/raw").glob("*.csv.gz"))
TABLE = "loans_raw"

header = pd.read_csv(CSV_PATH, nrows=0)
columns = [c.strip().lower() for c in header.columns]
print(f"{len(columns)} columns found")

column_defs = ",\n    ".join(f'"{c}" TEXT' for c in columns)
create_sql = f"CREATE TABLE {TABLE} (\n    {column_defs}\n);"

conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    dbname=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
)
cur = conn.cursor()

cur.execute(f"DROP TABLE IF EXISTS {TABLE};")
cur.execute(create_sql)
conn.commit()
print(f"table {TABLE} created")

start = time.time()
with gzip.open(CSV_PATH, "rt", encoding="utf-8", newline="") as f:
    cur.copy_expert(
        f"COPY {TABLE} FROM STDIN WITH (FORMAT csv, HEADER true)",
        f,
    )
conn.commit()
print(f"loaded in {time.time() - start:.1f} seconds")

cur.execute(f"SELECT COUNT(*) FROM {TABLE};")
print("rows in table:", cur.fetchone()[0])

cur.close()
conn.close()