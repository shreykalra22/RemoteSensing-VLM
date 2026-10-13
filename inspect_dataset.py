import pandas as pd

FILE = "data/raw/bigearthnet/BigEarthNet.txt.parquet"

print("=" * 60)
print("Loading BigEarthNet.txt metadata...")
print("=" * 60)

df = pd.read_parquet(FILE)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print(" -", column)

print("\nFirst 3 records:")
print(df.head(3).to_string())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("Inspection complete")
print("=" * 60)