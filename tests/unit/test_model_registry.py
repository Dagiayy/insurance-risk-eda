import pytest
from sklearn.linear_model import LinearRegression

from insurance_analytics.ml.registry import ModelMetadata, ModelRegistry


def test_register_and_load_roundtrip(tmp_path):
    registry = ModelRegistry(tmp_path)
    model = LinearRegression()
    metadata = ModelMetadata(model_name="demo", version="v1", created_at="v1", metrics={"rmse": 1.0}, feature_columns=["a", "b"])

    registry.register("demo", model, metadata)
    loaded_model, loaded_meta = registry.load("demo")

    assert isinstance(loaded_model, LinearRegression)
    assert loaded_meta.version == "v1"
    assert loaded_meta.feature_columns == ["a", "b"]


def test_load_without_version_uses_latest_pointer(tmp_path):
    registry = ModelRegistry(tmp_path)
    for version in ["v1", "v2"]:
        registry.register("demo", LinearRegression(), ModelMetadata(model_name="demo", version=version, created_at=version))

    _, meta = registry.load("demo")
    assert meta.version == "v2"


def test_load_missing_model_raises_file_not_found(tmp_path):
    registry = ModelRegistry(tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.load("nonexistent")


def test_has_model_reflects_registration_state(tmp_path):
    registry = ModelRegistry(tmp_path)
    assert registry.has_model("demo") is False
    registry.register("demo", LinearRegression(), ModelMetadata(model_name="demo", version="v1", created_at="v1"))
    assert registry.has_model("demo") is True
