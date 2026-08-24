import pandas as pd

from insurance_analytics.data.clean import clean_policies, profile_data_quality


def test_clean_policies_deduplicates_on_cover_id(raw_policy_df):
    df = pd.concat([raw_policy_df, raw_policy_df.iloc[:5]], ignore_index=True)
    cleaned, report = clean_policies(df, dedupe_on="UnderwrittenCoverID")
    assert report.duplicate_rows_dropped == 5
    assert len(cleaned) == len(raw_policy_df)
    assert cleaned["UnderwrittenCoverID"].is_unique


def test_clean_policies_fills_missing_total_claims_with_zero_not_median():
    df = pd.DataFrame({"TotalClaims": [100.0, None, 50.0], "UnderwrittenCoverID": [1, 2, 3]})
    cleaned, _ = clean_policies(df, dedupe_on="UnderwrittenCoverID")
    assert cleaned["TotalClaims"].iloc[1] == 0.0


def test_clean_policies_fills_categorical_unknown(raw_policy_df):
    cleaned, _ = clean_policies(raw_policy_df)
    assert cleaned["Gender"].isnull().sum() == 0
    assert "Unknown" in cleaned["Gender"].unique()


def test_clean_policies_report_reflects_row_counts(raw_policy_df):
    cleaned, report = clean_policies(raw_policy_df)
    assert report.input_rows == len(raw_policy_df)
    assert report.output_rows == len(cleaned)


def test_profile_data_quality_flags_iqr_outliers():
    df = pd.DataFrame({"TotalPremium": [100, 105, 98, 102, 10_000]})
    profile = profile_data_quality(df, numeric_cols=["TotalPremium"])
    assert profile["outliers_iqr"]["TotalPremium"] == 1
