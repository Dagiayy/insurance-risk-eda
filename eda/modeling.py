import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import shap
from pathlib import Path

# Create output directory
Path('docs').mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
try:
    df = pd.read_csv('data/insurance_data_cleaned.txt', sep='|')
except FileNotFoundError:
    print("Dataset not found. Run src/preprocess_data.py and dvc pull.")
    exit(1)

# Feature Engineering
df['VehicleAge'] = 2015 - df['RegistrationYear']
df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].clip(lower=0.01)
df['IsNewVehicle'] = (df['NewVehicle'] == 'New').astype(int)
df['PremiumPerSumInsured'] = df['TotalPremium'] / df['SumInsured'].clip(lower=1)
df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)

# Group rare categories
for col in ['VehicleType', 'CoverType']:
    counts = df[col].value_counts()
    rare = counts[counts < len(df) * 0.01].index
    df[col] = df[col].apply(lambda x: 'Other' if x in rare else x)

# Encode categorical variables
cat_cols = ['Province', 'VehicleType', 'CoverType', 'Gender', 'MaritalStatus']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
le = LabelEncoder()
df['PostalCode'] = le.fit_transform(df['PostalCode'])

# Features and targets
features = [col for col in df.columns if col not in ['TotalClaims', 'HasClaim', 'TotalPremium', 'NewVehicle', 'Bank', 'AccountType', 'TransactionMonth']]
X_severity = df[df['HasClaim'] == 1][features]
y_severity = df[df['HasClaim'] == 1]['TotalClaims']
X_prob = df[features]
y_prob = df['HasClaim']

# Train-test split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_severity, y_severity, test_size=0.2, random_state=42)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_prob, y_prob, test_size=0.2, random_state=42)

# Regression models for severity
models_s = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Classification models for probability
models_p = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# Evaluate regression models
results_s = []
for name, model in models_s.items():
    if name in ['Random Forest', 'XGBoost']:
        param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid.fit(X_train_s, y_train_s)
        model = grid.best_estimator_
    else:
        model.fit(X_train_s, y_train_s)
    y_pred = model.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(y_test_s, y_pred))
    r2 = r2_score(y_test_s, y_pred)
    results_s.append({'Model': name, 'RMSE': rmse, 'R-squared': r2})

# Evaluate classification models
results_p = []
for name, model in models_p.items():
    param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    grid = GridSearchCV(model, param_grid, cv=3, scoring='f1')
    grid.fit(X_train_p, y_train_p)
    model = grid.best_estimator_
    y_pred = model.predict(X_test_p)
    results_p.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test_p, y_pred),
        'Precision': precision_score(y_test_p, y_pred, zero_division=0),
        'Recall': recall_score(y_test_p, y_pred, zero_division=0),
        'F1-score': f1_score(y_test_p, y_pred, zero_division=0)
    })

# SHAP analysis for best model (XGBoost severity)
xgb_s = models_s['XGBoost']
explainer = shap.Explainer(xgb_s, X_train_s)
shap_values = explainer(X_test_s)
shap.summary_plot(shap_values, X_test_s, plot_type="bar", max_display=10, show=False)
plt.savefig('notebooks/plots/shap_summary_severity.png')
plt.close()

# Save results
results_s_df = pd.DataFrame(results_s)
results_p_df = pd.DataFrame(results_p)
results_s_df.to_csv('docs/severity_model_results.csv', index=False)
results_p_df.to_csv('docs/probability_model_results.csv', index=False)

# Premium calculation (example)
prob_pred = models_p['XGBoost'].predict_proba(X_test_p)[:, 1]
severity_pred = xgb_s.predict(X_test_p)
expense_loading = 0.1 * severity_pred
profit_margin = 0.15 * prob_pred * severity_pred
premium = prob_pred * severity_pred + expense_loading + profit_margin
premium_df = pd.DataFrame({'PredictedPremium': premium})
premium_df.to_csv('docs/predicted_premiums.csv', index=False)

print("Severity Model Results:")
print(results_s_df)
print("\nProbability Model Results:")
print(results_p_df)