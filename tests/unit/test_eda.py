from insurance_analytics.analysis.eda import compute_eda_summary, detect_outliers_iqr


def test_compute_eda_summary_contains_expected_keys(raw_policy_df):
    summary = compute_eda_summary(raw_policy_df)
    assert summary["n_rows"] == len(raw_policy_df)
    assert "overall_loss_ratio" in summary
    assert "claim_frequency" in summary
    assert 0.0 <= summary["claim_frequency"] <= 1.0


def test_detect_outliers_iqr_returns_subset(raw_policy_df):
    outliers = detect_outliers_iqr(raw_policy_df, "TotalClaims")
    assert (outliers.index.isin(raw_policy_df.index)).all()
