from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
csv_path = next(RAW_DIR.glob("*.csv.gz"))
print("Reading:", csv_path)

sample = pd.read_csv(csv_path, nrows=1000, low_memory=False)

print("columns:", sample.shape[1])
print()
for col in sample.columns:
    print(col)
print()
print(sample.head(3).T.head(20))