Predictive Modeling Report
Overview
This report details predictive models built for AlphaCare Insurance Solutions (ACIS) to support a risk-based pricing system. We developed models for claim severity (regression), claim probability (classification), and premium optimization using insurance_data_cleaned.txt (1,000,098 rows, 2,788 with claims).
Models and Results
Claim Severity (Regression)

Target: TotalClaims for policies with claims (> 0).
Metrics: RMSE, R-squared.
Results:
Linear Regression: RMSE = 33,512 Rand, R² = 0.302
Moderate fit, capturing 30% of variance.


Random Forest: RMSE = 36,402 Rand, R² = 0.176
Underperformed due to feature limitations.


XGBoost: RMSE = 38,231 Rand, R² = 0.091
Poorest fit, likely due to small dataset (2,788 rows).




Best Model: Linear Regression, though performance is modest.
Note: Removal of LossRatio reduced predictive power. Future models should explore scaled targets (e.g., log-transform) or new features.

Claim Probability (Classification)

Target: HasClaim (1 if TotalClaims > 0, 0 otherwise).
Metrics: Accuracy, Precision, Recall, F1-score, CV F1 Mean/Std.
Results:
Random Forest:
Accuracy: 0.902, Precision: 0.021, Recall: 0.717, F1-score: 0.041
CV F1 Mean: 0.039, Std: 0.001
High recall but low precision due to class imbalance (0.28% claims).


XGBoost:
Accuracy: 0.960, Precision: 0.027, Recall: 0.363, F1-score: 0.051
CV F1 Mean: 0.048, Std: 0.002
Slightly better F1-score, more robust.




Best Model: XGBoost, with higher F1-score but limited by imbalance.
Note: Low F1-scores reflect class imbalance. Threshold tuning or ensemble methods could improve precision.

Premium Optimization

Framework: Premium = (Probability * Severity) + 10% Expense Loading + 15% Profit Margin.
Output: docs/predicted_premiums.csv.
Limitation: Low probability F1-scores may affect premium reliability.

Feature Importance (SHAP Analysis)

Model: XGBoost (severity).
Top Features:
CalculatedPremiumPerTerm (Importance: 10,177): Higher premiums indicate riskier policies, increasing claim severity.
SumInsured (9,264): Larger insured amounts drive higher claims.
UnderwrittenCoverID (2,998): Policy-specific risk patterns.
PolicyID (2,615): Similar to CoverID, captures unique policy risks.
PremiumPerSumInsured (2,096): Lower ratios reduce severity.
PostalCode (1,815): Urban areas (e.g., Gauteng) have higher claims.
CustomValueEstimate (1,734): Vehicle value correlates with claim size.
RegistrationYear (1,354): Older vehicles (2015 - RegistrationYear) increase severity by ~1,500 Rand per year.
CapitalOutstanding (1,307): Higher loan amounts linked to larger claims.
CoverType_Windscreen (581): Specific coverage impacts severity.


Business Implications:
Pricing: Adjust premiums for high SumInsured, older vehicles, and urban areas.
Segmentation: Market to policies with low PremiumPerSumInsured.
Product Design: Reassess windscreen coverage pricing.



Methodology

Data: insurance_data_cleaned.txt.
Preparation:
Features: VehicleAge, IsNewVehicle, PremiumPerSumInsured.
Encoding: One-hot for categoricals, label for PostalCode.
Split: 80:20 train-test.


Models: Linear Regression, Random Forests, XGBoost.
Evaluation: RMSE, R² for regression; accuracy, precision, recall, F1 for classification.
Interpretability: SHAP for XGBoost.

Business Impact

Pricing: Modest severity predictions support basic risk-based pricing.
Segmentation: Target low-risk regions and newer vehicles.
Risk Management: Monitor high SumInsured policies.
Next Steps: Improve models with log-transformed targets, threshold tuning, or ensemble methods.

Outputs

Severity Results: docs/severity_model_results.csv
Probability Results: docs/probability_model_results.csv
SHAP Features: docs/shap_features.csv
Premiums: docs/predicted_premiums.csv
SHAP Plot: notebooks/plots/shap_summary_severity.png

