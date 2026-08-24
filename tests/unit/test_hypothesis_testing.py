from insurance_analytics.analysis.hypothesis_testing import GroupComparison, run_all_hypothesis_tests


def test_run_all_hypothesis_tests_returns_results_for_each_comparison(raw_policy_df):
    comparisons = [GroupComparison("Province", "Gauteng", "Western Cape", "Provinces")]
    results, equivalence = run_all_hypothesis_tests(raw_policy_df, comparisons)
    assert set(results["metric"]) == {"Claim Frequency", "Claim Severity"}
    assert not equivalence.empty


def test_significant_flag_matches_p_value_threshold(raw_policy_df):
    comparisons = [GroupComparison("Province", "Gauteng", "Western Cape", "Provinces")]
    results, _ = run_all_hypothesis_tests(raw_policy_df, comparisons)
    for _, row in results.iterrows():
        if row["p_value"] is not None:
            assert row["significant"] == (row["p_value"] < 0.05)
