import pandas as pd

from insurance_analytics.features.engineering import encode_categoricals, engineer_features


def test_vehicle_age_uses_configurable_reference_year_not_hardcoded_2015():
    df = pd.DataFrame({"RegistrationYear": [2010, 2005]})
    features_2015 = engineer_features(df, reference_year=2015)
    features_2026 = engineer_features(df, reference_year=2026)
    assert features_2015["VehicleAge"].tolist() == [5, 10]
    assert features_2026["VehicleAge"].tolist() == [16, 21]
    assert features_2026["VehicleAge"].tolist()[0] != features_2015["VehicleAge"].tolist()[0]


def test_vehicle_age_clipped_at_zero_for_future_registration_years():
    df = pd.DataFrame({"RegistrationYear": [2020]})
    features = engineer_features(df, reference_year=2015)
    assert features["VehicleAge"].iloc[0] == 0


def test_premium_per_sum_insured_avoids_division_by_zero():
    df = pd.DataFrame({"TotalPremium": [100.0], "SumInsured": [0.0]})
    features = engineer_features(df, reference_year=2015)
    assert features["PremiumPerSumInsured"].iloc[0] == 100.0  # clipped denominator to 1


def test_postal_code_frequency_encoding_replaces_label_encoding():
    df = pd.DataFrame({"PostalCode": [2000, 2000, 2000, 8000]})
    features = engineer_features(df, reference_year=2015)
    assert features["PostalCodeFrequency"].iloc[0] == 0.75
    assert features["PostalCodeFrequency"].iloc[3] == 0.25


def test_encode_categoricals_groups_rare_categories():
    df = pd.DataFrame({"Province": ["A"] * 95 + ["Rare"] * 5})
    encoded = encode_categoricals(df, categorical_cols=["Province"], rare_threshold=0.1)
    assert "Province_Rare" not in encoded.columns  # grouped into "Other" then dropped by drop_first or absent


def test_encode_categoricals_reindexes_to_fitted_columns_for_inference():
    train_df = pd.DataFrame({"Province": ["Gauteng", "Western Cape", "Gauteng"]})
    train_encoded = encode_categoricals(train_df, categorical_cols=["Province"])
    fitted_columns = list(train_encoded.columns)

    new_df = pd.DataFrame({"Province": ["Eastern Cape"]})  # unseen category at inference time
    new_encoded = encode_categoricals(new_df, categorical_cols=["Province"], fitted_columns=fitted_columns)

    assert list(new_encoded.columns) == fitted_columns
    assert new_encoded.shape[0] == 1
