# src/eda.py
import pandas as pd

# Load dataset (replace with actual path)
df = pd.read_csv('../data/insurance_data.csv')

# Ensure TransactionMonth is datetime
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

# Check data types
print("Data Types:")
print(df.dtypes)

# Descriptive stats for numeric features
num_cols = ['TotalPremium','TotalClaims','SumInsured','CustomValueEstimate']
print(df[num_cols].describe())

# Check missing values
print(df[num_cols + ['Province','VehicleType','Gender']].isna().sum())


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Histograms for numeric
for col in num_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'notebooks/plots/hist_{col}.png')
    plt.close()

# Bar charts for key categorical
for col in ['Province','VehicleType','Gender']:
    plt.figure()
    sns.countplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'notebooks/plots/bar_{col}.png')
    plt.close()
