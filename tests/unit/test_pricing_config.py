from insurance_analytics.config import PricingConfig


def test_calculate_premium_matches_documented_formula():
    pricing = PricingConfig(expense_loading_pct=0.10, profit_margin_pct=0.15)
    probability, severity = 0.2, 1000.0
    expected_loss = probability * severity
    expected = expected_loss + 0.10 * severity + 0.15 * expected_loss
    assert pricing.calculate_premium(probability, severity) == expected


def test_calculate_premium_is_zero_when_probability_and_severity_are_zero():
    pricing = PricingConfig()
    assert pricing.calculate_premium(0.0, 0.0) == 0.0


def test_pricing_config_overridable_via_env(monkeypatch):
    monkeypatch.setenv("PRICING_EXPENSE_LOADING_PCT", "0.25")
    pricing = PricingConfig()
    assert pricing.expense_loading_pct == 0.25
