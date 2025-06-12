import pandas as pd

df = pd.read_csv('data/insurance_data.csv')

print("Shape of the dataset:", df.shape)
print("Columns:", df.columns.tolist())

print("First 5 rows:\n", df.head())
