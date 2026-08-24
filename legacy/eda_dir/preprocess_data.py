import pandas as pd

# Load raw dataset
df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|')

# Imputation (from eda.py)
df['CustomValueEstimate'] = df['CustomValueEstimate'].fillna(df['CustomValueEstimate'].median())
df['TotalClaims'] = df['TotalClaims'].fillna(0)
df['Gender'] = df['Gender'].fillna('Unknown')
df['Province'] = df['Province'].fillna('Unknown')
df['Bank'] = df['Bank'].fillna('Unknown')
df['AccountType'] = df['AccountType'].fillna('Unknown')
df['MaritalStatus'] = df['MaritalStatus'].fillna('Unknown')
df['NewVehicle'] = df['NewVehicle'].fillna('Unknown')

# Save cleaned dataset
df.to_csv('data/MachineLearningRating_v3_cleaned.txt', sep='|', index=False)
print("Created data/MachineLearningRating_v3_cleaned.txt")