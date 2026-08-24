"""Pydantic request/response models for the pricing API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PolicyRequest(BaseModel):
    """A single policy record to score. Field names match the canonical schema."""

    Province: str = Field(examples=["Gauteng"])
    PostalCode: str = Field(examples=["2000"])
    Gender: str = Field(examples=["Male"])
    MaritalStatus: str = Field(default="Unknown")
    VehicleType: str = Field(examples=["Passenger Vehicle"])
    CoverType: str = Field(examples=["Comprehensive"])
    RegistrationYear: int = Field(ge=1950, le=2035, examples=[2015])
    NewVehicle: str = Field(default="Unknown")
    SumInsured: float = Field(ge=0, examples=[250000.0])
    CustomValueEstimate: float = Field(ge=0, examples=[220000.0])
    TotalPremium: float = Field(ge=0, examples=[1200.0])

    model_config = {"extra": "allow"}  # tolerate additional raw-data columns without failing


class PricingResponse(BaseModel):
    claim_probability: float
    claim_severity: float
    predicted_premium: float
    model_versions: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool


class InsightRequest(BaseModel):
    stats: dict = Field(description="Pre-computed EDA/hypothesis-test summary, e.g. from /analysis/eda")


class ErrorResponse(BaseModel):
    detail: str
