import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plot style
sns.set(style="whitegrid")

# Create plots directory
Path('../notebooks/plots').mkdir(parents=True, exist_ok=True)

# Load dataset (replace with actual path)
try:
    df = pd.read_csv('../data/insurance_data.csv')
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


# Average TotalPremium by Province and CoverType
premium_by_province_cover = df.groupby(['Province', 'CoverType'])['TotalPremium'].mean().unstack()
print("\nAverage TotalPremium by Province and CoverType:")
print(premium_by_province_cover)

# Top vehicle makes by Province
print("\nTop Vehicle Makes by Province:")
for province in df['Province'].unique():
    top_makes = df[df['Province'] == province]['Make'].value_counts().head(3)
    print(f"{province}: {top_makes}")


# Box plots for outlier detection
for col in ['TotalClaims', 'CustomValueEstimate']:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f'Box Plot of {col}')
    plt.savefig(f'notebooks/plots/box_{col}.png')
    plt.close()

# Identify outliers (IQR method)
for col in ['TotalClaims', 'CustomValueEstimate']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
    print(f"\nOutliers in {col}: {len(outliers)}")


# Claim frequency (proportion of policies with claims)
claim_freq = df.groupby('TransactionMonth').apply(lambda x: (x['TotalClaims'] > 0).mean())
# Claim severity (average claim amount where claims > 0)
claim_severity = df[df['TotalClaims'] > 0].groupby('TransactionMonth')['TotalClaims'].mean()

print("\nClaim Frequency by Month:")
print(claim_freq)
print("\nClaim Severity by Month:")
print(claim_severity)


# Average TotalClaims by Make (for policies with claims)
claims_by_make = df[df['TotalClaims'] > 0].groupby('Make')['TotalClaims'].mean().sort_values()
print("\nTop 5 Makes with Highest Claim Amounts:")
print(claims_by_make.tail(5))
print("\nTop 5 Makes with Lowest Claim Amounts:")
print(claims_by_make.head(5))



# Visualization 1: Loss Ratio by Province
plt.figure(figsize=(10, 6))
loss_by_province = df.groupby('Province')['LossRatio'].mean().sort_values()
sns.barplot(x=loss_by_province.index, y=loss_by_province.values, palette='magma')
plt.title('Average Loss Ratio by Province', fontsize=14)
plt.xlabel('Province', fontsize=12)
plt.ylabel('Loss Ratio (Claims/Premium)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('notebooks/plots/loss_ratio_province.png')
plt.close()


# Visualization 2: Claim Severity by Vehicle Make (Top 10)
top_makes = df['Make'].value_counts().index[:10]
claims_by_make = df[df['TotalClaims'] > 0][df['Make'].isin(top_makes)]
plt.figure(figsize=(12, 6))
sns.boxplot(x='Make', y='TotalClaims', data=claims_by_make, palette='Set2')
plt.title('Claim Severity by Vehicle Make (Top 10)', fontsize=14)
plt.xlabel('Vehicle Make', fontsize=12)
plt.ylabel('Total Claims (Rand)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('notebooks/plots/claim_severity_make.png')
plt.close()



# Visualization 3: Temporal Trend of Claim Frequency and Severity
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(claim_freq.index, claim_freq.values, marker='o', color='blue', label='Claim Frequency')
ax1.set_xlabel('Transaction Month', fontsize=12)
ax1.set_ylabel('Claim Frequency', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticklabels(claim_freq.index, rotation=45)

ax2 = ax1.twinx()
ax2.plot(claim_severity.index, claim_severity.values, marker='s', color='red', label='Claim Severity')
ax2.set_ylabel('Claim Severity (Rand)', color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Claim Frequency and Severity Over Time', fontsize=14)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.tight_layout()
plt.savefig('notebooks/plots/temporal_trends.png')
plt.close()