# src/eda.py
import pandas as pd

# Load dataset (replace with actual path)
df = pd.read_csv('../data/insurance_data.csv')

# Ensure TransactionMonth is datetime
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

# Check data types
print("Data Types:")
print(df.dtypes)