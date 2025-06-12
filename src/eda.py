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


# Calculate Loss Ratio
df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, 1e-10)  # Avoid division by zero

# Loss Ratio by Province, VehicleType, Gender
for group in ['Province', 'VehicleType', 'Gender']:
    print(f"\nLoss Ratio by {group}:")
    print(df.groupby(group)['LossRatio'].mean().sort_values())

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('notebooks/plots/correlation_matrix.png')
plt.close()

# Scatter plot: TotalPremium vs TotalClaims by PostalCode (top 5)
top_zips = df['PostalCode'].value_counts().index[:5]
plt.figure(figsize=(10, 6))
for zipcode in top_zips:
    subset = df[df['PostalCode'] == zipcode]
    plt.scatter(subset['TotalPremium'], subset['TotalClaims'], label=zipcode, alpha=0.6)
plt.xlabel('Total Premium (Rand)')
plt.ylabel('Total Claims (Rand)')
plt.title('Total Premium vs Total Claims by PostalCode')
plt.legend()
plt.savefig('notebooks/plots/scatter_premium_claims_zip.png')
plt.close()