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