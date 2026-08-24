import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
from pathlib import Path

# Create output directory
Path('docs').mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
try:
    df = pd.read_csv('data/insurance_data_cleaned.txt', sep='|')
except FileNotFoundError:
    print("Dataset not found. Please run src/preprocess_data.py and dvc pull.")
    exit(1)

# Compute KPIs
df['HasClaim'] = df['TotalClaims'] > 0
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

# Function to perform chi-squared test for Claim Frequency
def test_claim_frequency(group_col, group_a_val, group_b_val, group_name):
    group_a = df[df[group_col] == group_a_val]['HasClaim']
    group_b = df[df[group_col] == group_b_val]['HasClaim']
    contingency_table = pd.crosstab(df[df[group_col].isin([group_a_val, group_b_val])][group_col], df['HasClaim'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    return {'Metric': 'Claim Frequency', 'Group': group_name, 'p-value': p,
            'Statistic': chi2, 'Group A Freq': group_a.mean(), 'Group B Freq': group_b.mean()}

# Function to perform t-test for Claim Severity
def test_claim_severity(group_col, group_a_val, group_b_val, group_name):
    group_a = df[(df[group_col] == group_a_val) & (df['HasClaim'])]['TotalClaims']
    group_b = df[(df[group_col] == group_b_val) & (df['HasClaim'])]['TotalClaims']
    if len(group_a) > 1 and len(group_b) > 1:
        t_stat, p = ttest_ind(group_a, group_b, equal_var=False)
        return {'Metric': 'Claim Severity', 'Group': group_name, 'p-value': p,
                'Statistic': t_stat, 'Group A Mean': group_a.mean(), 'Group B Mean': group_b.mean()}
    return {'Metric': 'Claim Severity', 'Group': group_name, 'p-value': np.nan,
            'Statistic': np.nan, 'Group A Mean': group_a.mean(), 'Group B Mean': group_b.mean()}

# Function to perform t-test for Margin
def test_margin(group_col, group_a_val, group_b_val, group_name):
    group_a = df[df[group_col] == group_a_val]['Margin']
    group_b = df[df[group_col] == group_b_val]['Margin']
    t_stat, p = ttest_ind(group_a, group_b, equal_var=False)
    return {'Metric': 'Margin', 'Group': group_name, 'p-value': p,
            'Statistic': t_stat, 'Group A Mean': group_a.mean(), 'Group B Mean': group_b.mean()}

# Equivalence check for group attributes
def check_equivalence(group_col, group_a_val, group_b_val, attributes):
    results = []
    for attr in attributes:
        if df[attr].dtype in ['object', 'category']:
            table = pd.crosstab(df[df[group_col].isin([group_a_val, group_b_val])][group_col], df[attr])
            if table.shape[0] == 2 and table.shape[1] > 0:
                chi2, p, _, _ = chi2_contingency(table)
                results.append({'Attribute': attr, 'Test': 'Chi-squared', 'p-value': p})
        else:
            group_a = df[df[group_col] == group_a_val][attr].dropna()
            group_b = df[df[group_col] == group_b_val][attr].dropna()
            if len(group_a) > 1 and len(group_b) > 1:
                t_stat, p = ttest_ind(group_a, group_b, equal_var=False)
                results.append({'Attribute': attr, 'Test': 'T-test', 'p-value': p})
    return pd.DataFrame(results)

# Define tests
tests = [
    {'group_col': 'Province', 'group_a_val': 'Gauteng', 'group_b_val': 'Western Cape', 'group_name': 'Provinces'},
    {'group_col': 'PostalCode', 'group_a_val': 2000, 'group_b_val': 8000, 'group_name': 'Zip Codes (Risk)'},
    {'group_col': 'PostalCode', 'group_a_val': 2000, 'group_b_val': 8000, 'group_name': 'Zip Codes (Margin)'},
    {'group_col': 'Gender', 'group_a_val': 'Male', 'group_b_val': 'Female', 'group_name': 'Gender'}
]

# Attributes to check for equivalence
attributes = ['VehicleType', 'RegistrationYear', 'CoverType', 'SumInsured']

# Run tests and equivalence checks
results = []
equivalence_results = []
for test in tests:
    equiv_df = check_equivalence(test['group_col'], test['group_a_val'], test['group_b_val'], attributes)
    equiv_df['Group'] = test['group_name']
    equivalence_results.append(equiv_df)

    if test['group_name'] == 'Provinces' or test['group_name'] == 'Gender':
        results.append(test_claim_frequency(test['group_col'], test['group_a_val'], test['group_b_val'], test['group_name']))
        results.append(test_claim_severity(test['group_col'], test['group_a_val'], test['group_b_val'], test['group_name']))
    elif test['group_name'] == 'Zip Codes (Risk)':
        results.append(test_claim_frequency(test['group_col'], test['group_a_val'], test['group_b_val'], test['group_name']))
        results.append(test_claim_severity(test['group_col'], test['group_a_val'], test['group_b_val'], test['group_name']))
    elif test['group_name'] == 'Zip Codes (Margin)':
        results.append(test_margin(test['group_col'], test['group_a_val'], test['group_b_val'], test['group_name']))

# Save results
results_df = pd.DataFrame(results)
equivalence_df = pd.concat(equivalence_results)
results_df.to_csv('docs/hypothesis_test_results.csv', index=False)
equivalence_df.to_csv('docs/equivalence_check_results.csv', index=False)

# Print results
print("Hypothesis Test Results:")
print(results_df)
print("\nEquivalence Check Results:")
print(equivalence_df)