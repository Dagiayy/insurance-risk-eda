import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score
)
import shap
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
Path('docs').mkdir(parents=True, exist_ok=True)
Path('notebooks/plots').mkdir(parents=True, exist_ok=True)

# Load dataset
try:
    df = pd.read_csv('data/MachineLearningRating_v3_cleaned.txt', sep='|', low_memory=False)
    logging.info("Dataset loaded successfully")
except FileNotFoundError:
    logging.error("Dataset not found. Run src/preprocess_data.py and dvc pull.")
    exit(1)

# Coerce numeric fields
numeric_cols = ['TotalClaims', 'TotalPremium', 'SumInsured', 'RegistrationYear']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=numeric_cols, inplace=True)
logging.info(f"Rows after dropping NA in numeric cols: {len(df)}")

# Feature engineering
df['VehicleAge'] = 2015 - df['RegistrationYear']
df['IsNewVehicle'] = (df['NewVehicle'] == 'New').astype(int)
df['PremiumPerSumInsured'] = df['TotalPremium'] / df['SumInsured'].clip(lower=1)
df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)

# Exclude LossRatio to avoid data leakage
# df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].clip(lower=0.01)

# Group rare categories
cat_cols = ['Province', 'VehicleType', 'CoverType', 'Gender', 'MaritalStatus']
for col in cat_cols:
    if col in df.columns:
        counts = df[col].value_counts()
        rare = counts[counts < len(df) * 0.01].index
        df[col] = df[col].apply(lambda x: 'Other' if x in rare else x)
    else:
        df[col] = 'Unknown'

# Fill missing categorical columns
for col in cat_cols:
    df[col] = df[col].fillna('Unknown')

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
logging.info(f"Columns after one-hot encoding: {list(df.columns)}")

# Encode PostalCode
if 'PostalCode' in df.columns:
    le = LabelEncoder()
    df['PostalCode'] = le.fit_transform(df['PostalCode'].astype(str))

# Drop unnecessary columns
drop_cols = ['TotalClaims', 'HasClaim', 'TotalPremium', 'NewVehicle', 'Bank', 'AccountType', 'TransactionMonth', 'LossRatio']
features = [col for col in df.columns if col not in drop_cols]
logging.info(f"Selected features: {features}")

# Ensure all features are numeric
X_severity = df[df['HasClaim'] == 1][features].apply(pd.to_numeric, errors='coerce')
y_severity = np.log1p(df[df['HasClaim'] == 1]['TotalClaims'])
X_prob = df[features].apply(pd.to_numeric, errors='coerce')
y_prob = df['HasClaim']

# Drop columns with NaN after coercion
X_severity = X_severity.dropna(axis=1)
X_prob = X_prob[X_severity.columns]  # Align columns
X_severity.dropna(inplace=True)
X_prob.dropna(inplace=True)
y_severity = y_severity.loc[X_severity.index]
y_prob = y_prob.loc[X_prob.index]
logging.info(f"Severity data shape: {X_severity.shape}, Probability data shape: {X_prob.shape}")

# Train-test splits
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_severity, y_severity, test_size=0.2, random_state=42)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_prob, y_prob, test_size=0.2, random_state=42)
logging.info(f"Train/test split shapes: Severity {X_train_s.shape}/{X_test_s.shape}, Probability {X_train_p.shape}/{X_test_p.shape}")

# Define models
models_s = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

models_p = {
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=len(y_train_p[y_train_p == 0]) / len(y_train_p[y_train_p == 1]))
}

# Regression model evaluation
results_s = []
for name, model in models_s.items():
    logging.info(f"Training severity model: {name}")
    try:
        if name in ['Random Forest', 'XGBoost']:
            param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
            grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid.fit(X_train_s, y_train_s)
            model = grid.best_estimator_
            logging.info(f"Best params for {name}: {grid.best_params_}")
        else:
            model.fit(X_train_s, y_train_s)
        y_pred = model.predict(X_test_s)
        results_s.append({
            'Model': name,
            'RMSE': np.sqrt(mean_squared_error(y_test_s, y_pred)),
            'R-squared': r2_score(y_test_s, y_pred)
        })
    except Exception as e:
        logging.error(f"{name} failed: {e}")

# Classification model evaluation with cross-validation
results_p = []
for name, model in models_p.items():
    logging.info(f"Training probability model: {name}")
    try:
        if name in ['Random Forest', 'XGBoost']:
            param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
            grid = GridSearchCV(model, param_grid, cv=3, scoring='f1')
            grid.fit(X_train_p, y_train_p)
            model = grid.best_estimator_
            logging.info(f"Best params for {name}: {grid.best_params_}")
        else:
            model.fit(X_train_p, y_train_p)
        y_pred = model.predict(X_test_p)
        cv_scores = cross_val_score(model, X_train_p, y_train_p, cv=5, scoring='f1')
        results_p.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test_p, y_pred),
            'Precision': precision_score(y_test_p, y_pred, zero_division=0),
            'Recall': recall_score(y_test_p, y_pred, zero_division=0),
            'F1-score': f1_score(y_test_p, y_pred, zero_division=0),
            'CV F1 Mean': cv_scores.mean(),
            'CV F1 Std': cv_scores.std()
        })
        models_p[name] = model  # Store fitted model
    except Exception as e:
        logging.error(f"{name} failed: {e}")

# SHAP analysis for severity
logging.info("Generating SHAP summary plot...")
xgb_s = models_s.get('XGBoost')
shap_features = []
if xgb_s:
    try:
        xgb_s.fit(X_train_s, y_train_s)
        explainer = shap.TreeExplainer(xgb_s)
        shap_values = explainer.shap_values(X_test_s)
        shap.summary_plot(shap_values, X_test_s, plot_type="bar", max_display=10, show=False)
        plt.savefig('notebooks/plots/shap_summary_severity.png')
        plt.close()
        # Extract top features
        shap_importance = pd.DataFrame({
            'Feature': X_test_s.columns,
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False).head(10)
        shap_features = shap_importance.to_dict('records')
        logging.info(f"Top SHAP features: {shap_features}")
        logging.info("SHAP plot generated successfully")
    except Exception as e:
        logging.error(f"SHAP analysis failed: {e}")

# Save results
results_s_df = pd.DataFrame(results_s)
results_p_df = pd.DataFrame(results_p)
results_s_df.to_csv('docs/severity_model_results.csv', index=False)
results_p_df.to_csv('docs/probability_model_results.csv', index=False)
logging.info("Model results saved to docs/")

# Calculate premiums
if 'XGBoost' in models_p and xgb_s:
    try:
        prob_pred = models_p['XGBoost'].predict_proba(X_test_p)[:, 1]
        severity_pred = xgb_s.predict(X_test_p)
        premium = prob_pred * severity_pred + 0.1 * severity_pred + 0.15 * prob_pred * severity_pred
        pd.DataFrame({'PredictedPremium': premium}).to_csv('docs/predicted_premiums.csv', index=False)
        logging.info("Premiums calculated and saved")
    except Exception as e:
        logging.error(f"Premium calculation failed: {e}")

# Save SHAP features
if shap_features:
    pd.DataFrame(shap_features).to_csv('docs/shap_features.csv', index=False)

# Final output
print("Severity Model Results:")
print(results_s_df)
print("\nProbability Model Results:")
print(results_p_df)