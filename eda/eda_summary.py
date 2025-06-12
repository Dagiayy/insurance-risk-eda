import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plot style
sns.set(style="whitegrid")

# Create directories
Path('notebooks/plots').mkdir(parents=True, exist_ok=True)
Path('results').mkdir(parents=True, exist_ok=True)

# Load pipe-separated dataset
try:
    df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|')
except FileNotFoundError:
    print("Dataset not found. Please provide insurance_data.txt")
    exit(1)

# Open a file to save results
with open('results/eda_results.txt', 'w') as f:
    def print_and_save(*args, **kwargs):
        """Helper function to print to terminal and save to file."""
        print(*args, **kwargs)  # Print to terminal
        print(*args, file=f, **kwargs)  # Save to file

    # Convert TransactionMonth to datetime
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

    # Check data types
    print_and_save("\n=== Data Types ===")
    print_and_save(df.dtypes)

    # Numerical columns for analysis
    numerical_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate', 'cubiccapacity', 'kilowatts', 'NumberOfDoors', 'Cylinders']

    # Descriptive statistics
    print_and_save("\n=== Descriptive Statistics ===")
    print_and_save(df[numerical_cols].describe().to_string())

    # Missing values
    print_and_save("\n=== Missing Values ===")
    print_and_save(df.isnull().sum().to_string())

    # Check duplicates in UnderwrittenCoverID
    print_and_save("\n=== Duplicate UnderwrittenCoverID ===")
    print_and_save(f"Total Duplicates: {df['UnderwrittenCoverID'].duplicated().sum()}")

    # Imputation
    df['CustomValueEstimate'] = df['CustomValueEstimate'].fillna(df['CustomValueEstimate'].median())
    df['TotalClaims'] = df['TotalClaims'].fillna(0)
    df['Gender'] = df['Gender'].fillna('Unknown')
    df['Province'] = df['Province'].fillna('Unknown')

    # Univariate Analysis: Histograms for numerical columns
    print_and_save("\n=== Univariate Analysis: Numerical Variables ===")
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), bins=30, kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'notebooks/plots/hist_{col}.png')
        plt.close()

    # Univariate Analysis: Bar charts for categorical columns
    categorical_cols = ['Province', 'VehicleType', 'Gender', 'CoverType', 'make']
    print_and_save("\n=== Univariate Analysis: Categorical Variables ===")
    for col in categorical_cols:
        top_categories = df[col].value_counts().head(20)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_categories.index, y=top_categories.values, palette='viridis')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'notebooks/plots/bar_{col}.png')
        plt.close()

    # Calculate Loss Ratio
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, 1e-10)

    # Loss Ratio by make and CoverType
    print_and_save("\n=== Loss Ratio Analysis ===")
    for group in ['make', 'CoverType']:
        loss_ratio = df.groupby(group)['LossRatio'].mean().sort_values()
        print_and_save(f"\nLoss Ratio by {group}:")
        print_and_save(loss_ratio.to_string())

    # Correlation matrix
    print_and_save("\n=== Correlation Matrix ===")
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('notebooks/plots/correlation_matrix.png')
    plt.close()

    # Scatter plot: TotalPremium vs TotalClaims by PostalCode
    print_and_save("\n=== Scatter Plot: TotalPremium vs TotalClaims by PostalCode ===")
    top_zips = df['PostalCode'].value_counts().index[:5]
    plt.figure(figsize=(10, 6))
    for zipcode in top_zips:
        subset = df[df['PostalCode'] == zipcode]
        plt.scatter(subset['TotalPremium'], subset['TotalClaims'], label=zipcode, alpha=0.6)
    plt.xlabel('Total Premium (Rand)')
    plt.ylabel('Total Claims (Rand)')
    plt.title('Total Premium vs Total Claims by PostalCode')
    plt.legend(title='PostalCode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('notebooks/plots/scatter_premium_claims_zip.png', bbox_inches='tight')
    plt.close()

    # Average TotalPremium by MainCrestaZone and CoverType
    premium_by_zone_cover = df.groupby(['MainCrestaZone', 'CoverType'])['TotalPremium'].mean().unstack()
    print_and_save("\n=== Average TotalPremium by MainCrestaZone and CoverType ===")
    print_and_save(premium_by_zone_cover.to_string())

    # Top vehicle makes by MainCrestaZone
    print_and_save("\n=== Top Vehicle Makes by MainCrestaZone ===")
    for zone in df['MainCrestaZone'].unique():
        top_makes = df[df['MainCrestaZone'] == zone]['make'].value_counts().head(3)
        print_and_save(f"{zone}: {top_makes.to_string()}")

    # Box plots for outlier detection
    print_and_save("\n=== Outlier Detection ===")
    for col in ['TotalPremium', 'CustomValueEstimate']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
        print_and_save(f"\nOutliers in {col}: {len(outliers)}")
        
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col], color='lightcoral')
        plt.title(f'Box Plot of {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'notebooks/plots/box_{col}.png')
        plt.close()

    # Claim frequency and severity
    claim_freq = df.groupby('TransactionMonth').apply(lambda x: (x['TotalClaims'] > 0).mean())
    claim_severity = df[df['TotalClaims'] > 0].groupby('TransactionMonth')['TotalClaims'].mean()

    print_and_save("\n=== Claim Frequency by Month ===")
    print_and_save(claim_freq.to_string())

    print_and_save("\n=== Claim Severity by Month ===")
    print_and_save(claim_severity.to_string())

    # Average TotalClaims by Make
    claims_by_make = df[df['TotalClaims'] > 0].groupby('make')['TotalClaims'].mean().sort_values()
    print_and_save("\n=== Top 5 Makes with Highest Claim Amounts ===")
    print_and_save(claims_by_make.tail(5).to_string())

    print_and_save("\n=== Top 5 Makes with Lowest Claim Amounts ===")
    print_and_save(claims_by_make.head(5).to_string())

    # Visualization 1: Average TotalPremium by CoverType
    print_and_save("\n=== Visualization: Average TotalPremium by CoverType ===")
    premium_by_cover = df.groupby('CoverType')['TotalPremium'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=premium_by_cover.index, y=premium_by_cover.values, palette='magma')
    plt.title('Average Total Premium by Cover Type', fontsize=14)
    plt.xlabel('Cover Type', fontsize=12)
    plt.ylabel('Average Total Premium (Rand)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('notebooks/plots/premium_by_covertype.png')
    plt.close()

    # Visualization 2: TotalPremium by Vehicle Make
    print_and_save("\n=== Visualization: TotalPremium by Vehicle Make ===")
    top_makes = df['make'].value_counts().index[:10]
    subset = df[df['make'].isin(top_makes)]
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='make', y='TotalPremium', data=subset, palette='Set2')
    plt.title('Total Premium by Vehicle Make', fontsize=14)
    plt.xlabel('Vehicle Make', fontsize=12)
    plt.ylabel('Total Premium (Rand)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('notebooks/plots/premium_by_make.png')
    plt.close()

    # Visualization 3: Temporal Trend of TotalPremium
    print_and_save("\n=== Visualization: Temporal Trend of TotalPremium ===")
    premium_trend = df.groupby('TransactionMonth')['TotalPremium'].mean()
    plt.figure(figsize=(10, 6))
    premium_trend.plot(kind='line', marker='o', color='blue')
    plt.title('Average Total Premium Over Time', fontsize=14)
    plt.xlabel('Transaction Month', fontsize=12)
    plt.ylabel('Average Total Premium (Rand)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('notebooks/plots/temporal_premium_trend.png')
    plt.close()
