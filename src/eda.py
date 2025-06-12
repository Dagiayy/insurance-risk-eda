import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plot style
sns.set(style="whitegrid")

# Create plots directory
Path('notebooks/plots').mkdir(parents=True, exist_ok=True)

# Load dataset (replace with actual path)
try:
    df = pd.read_csv('data/insurance_data.csv')
except FileNotFoundError:
    print("Dataset not found. Please provide insurance_data.csv")
    exit(1)

# Convert TransactionMonth to datetime
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

# Check data types
print("Data Types:")
print(df.dtypes)

# Numerical columns for analysis
numerical_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate', 'Cubiccapacity', 'Kilowatts']

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df[numerical_cols].describe())


# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicates in UnderwrittenCoverID
print("\nDuplicate UnderwrittenCoverID:", df['UnderwrittenCoverID'].duplicated().sum())

# Basic imputation (to be refined after data inspection)
df['TotalPremium'] = df['TotalPremium'].fillna(df['TotalPremium'].median())
df['TotalClaims'] = df['TotalClaims'].fillna(0)  # Assume no claim if missing
df['Gender'] = df['Gender'].fillna('Unknown')
df['Province'] = df['Province'].fillna('Unknown')


# Univariate analysis
# Histograms for numerical columns
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col].dropna(), bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f'notebooks/plots/hist_{col}.png')
    plt.close()

# Bar charts for categorical columns
categorical_cols = ['Province', 'VehicleType', 'Gender', 'CoverType']
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=col, palette='viridis')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'notebooks/plots/bar_{col}.png')
    plt.close()