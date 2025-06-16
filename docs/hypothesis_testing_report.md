Hypothesis Testing Report
Overview
This report summarizes A/B hypothesis testing to validate risk drivers for AlphaCare Insurance Solutions (ACIS). We tested four null hypotheses regarding risk (Claim Frequency, Claim Severity) and margin (TotalPremium - TotalClaims) across provinces, zip codes, and gender, using insurance_data_cleaned.txt.
Hypotheses and Results

H₀₁: No risk differences across provinces (Gauteng vs. Western Cape):
Claim Frequency: p = 6.93e-14, reject H₀. Gauteng has 55% higher claim frequency (0.336% vs. 0.217%).
Claim Severity: p = 0.0306, reject H₀. Western Cape claims are 26% higher (28,096 vs. 22,244 Rand).
Business Recommendation: Increase premiums in Gauteng by 10-15% to cover higher claim frequency. In Western Cape, consider a 5-10% premium adjustment or risk mitigation programs to address higher severity.


H₀₂: No risk differences between zip codes (2000 vs. 8000):
Claim Frequency: p = 0.2741, fail to reject H₀. No significant difference (0.364% vs. 0.432%).
Claim Severity: p = 0.0868, fail to reject H₀. Zip code 8000 claims are 75% higher (33,685 vs. 19,196 Rand).
Business Recommendation: Monitor claim severity in zip code 8000; consider a small premium adjustment if trend persists. Maintain current pricing for now.


H₀₃: No significant margin difference between zip codes (2000 vs. 8000):
Margin: p = 0.8315, fail to reject H₀. No significant difference (-8.11 vs. -16.04 Rand).
Business Recommendation: Address negative margins in both zip codes by reviewing pricing models to ensure profitability.


H₀₄: No significant risk difference between women and men:
Claim Frequency: p = 0.9515, fail to reject H₀. No significant difference (0.220% vs. 0.207%).
Claim Severity: p = 0.5680, fail to reject H₀. Female claims are 20% higher (17,875 vs. 14,859 Rand).
Business Recommendation: Maintain gender-neutral pricing, as risk profiles are similar. Monitor female claim severity for future segmentation.



Equivalence Checks

Most attributes (VehicleType, RegistrationYear, CoverType) showed significant differences (p < 0.05) between groups, except SumInsured for Provinces and Gender (p > 0.05).
Limitation: Group differences in attributes may confound results. Future analyses should subsample data to balance attributes or use regression to control for confounders.

Methodology

Metrics:
Claim Frequency: Proportion with claims.
Claim Severity: Average claim amount for claims > 0.
Margin: TotalPremium - TotalClaims.


Tests:
Chi-squared for Claim Frequency.
Welch’s t-test for Claim Severity and Margin.


Threshold: p < 0.05.
Data: insurance_data_cleaned.txt with imputed missing values.

Business Impact

Segmentation: Provincial differences support region-specific pricing adjustments.
Profitability: Negative margins highlight the need for pricing model review.
Fairness: Gender-neutral pricing aligns with similar risk profiles.

Outputs

Results: docs/hypothesis_test_results.csv
Equivalence Checks: docs/equivalence_check_results.csv

