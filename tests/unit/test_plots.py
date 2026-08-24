"""Smoke tests for headless plot generation (Agg backend, no display required)."""

from insurance_analytics.analysis.eda import correlation_matrix
from insurance_analytics.analysis.plots import save_bar_by_group, save_correlation_heatmap, save_histogram


def test_save_histogram_writes_a_png(raw_policy_df, tmp_path):
    out_path = save_histogram(raw_policy_df, "TotalPremium", tmp_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_correlation_heatmap_writes_a_png(raw_policy_df, tmp_path):
    corr = correlation_matrix(raw_policy_df, columns=["TotalPremium", "TotalClaims", "SumInsured"])
    out_path = save_correlation_heatmap(corr, tmp_path)
    assert out_path.exists()


def test_save_bar_by_group_writes_a_png(raw_policy_df, tmp_path):
    series = raw_policy_df.groupby("Province")["TotalPremium"].mean()
    out_path = save_bar_by_group(series, "Avg Premium by Province", "Premium", tmp_path, "bar.png")
    assert out_path.exists()
