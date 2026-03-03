"""
inference_api/schemas.py

Pydantic request/response models for the inference API.

Why Pydantic validation here?
    - Rejects malformed inputs BEFORE they touch the model.
    - Auto-generates OpenAPI docs at /docs.
    - Field validators enforce domain rules (e.g. tenure >= 0).

Production note:
    If the feature set changes (new columns added), update ChurnFeatures
    and bump the API version (v2). Never silently drop fields—that causes
    silent mispredictions which are worse than errors.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


# ── Enumerations (match training data values exactly) ─────────────────────────

class GenderEnum(str, Enum):
    male   = "Male"
    female = "Female"


class YesNoEnum(str, Enum):
    yes = "Yes"
    no  = "No"


class MultipleLineEnum(str, Enum):
    yes               = "Yes"
    no                = "No"
    no_phone_service  = "No phone service"


class InternetServiceEnum(str, Enum):
    dsl         = "DSL"
    fiber_optic = "Fiber optic"
    no          = "No"


class InternetAddonEnum(str, Enum):
    yes                  = "Yes"
    no                   = "No"
    no_internet_service  = "No internet service"


class ContractEnum(str, Enum):
    month_to_month = "Month-to-month"
    one_year       = "One year"
    two_year       = "Two year"


class PaymentMethodEnum(str, Enum):
    electronic_check = "Electronic check"
    mailed_check     = "Mailed check"
    bank_transfer    = "Bank transfer (automatic)"
    credit_card      = "Credit card (automatic)"


# ── Request model ──────────────────────────────────────────────────────────────

class ChurnFeatures(BaseModel):
    """
    Input features for a single churn prediction request.
    Field names match the Telco dataset columns exactly (after dropping customerID).
    """

    gender:           GenderEnum
    SeniorCitizen:    Annotated[int, Field(ge=0, le=1)]
    Partner:          YesNoEnum
    Dependents:       YesNoEnum
    tenure:           Annotated[int, Field(ge=0, le=100, description="Months as customer")]
    PhoneService:     YesNoEnum
    MultipleLines:    MultipleLineEnum
    InternetService:  InternetServiceEnum
    OnlineSecurity:   InternetAddonEnum
    OnlineBackup:     InternetAddonEnum
    DeviceProtection: InternetAddonEnum
    TechSupport:      InternetAddonEnum
    StreamingTV:      InternetAddonEnum
    StreamingMovies:  InternetAddonEnum
    Contract:         ContractEnum
    PaperlessBilling: YesNoEnum
    PaymentMethod:    PaymentMethodEnum
    MonthlyCharges:   Annotated[float, Field(ge=0.0, le=200.0)]
    TotalCharges:     Annotated[float, Field(ge=0.0)]

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.35,
                "TotalCharges": 904.20,
            }
        }
    }


# ── Response models ────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Prediction result returned to the caller."""
    prediction:       int   = Field(description="0 = No churn, 1 = Churn")
    probability:      float = Field(description="Probability of churn (0.0 – 1.0)")
    model_version:    str   = Field(description="MLflow model version used")
    model_stage:      str   = Field(description="MLflow stage (Production)")
    threshold:        float = Field(description="Decision threshold applied")


class HealthResponse(BaseModel):
    """Health check response."""
    status:        str
    model_loaded:  bool
    model_version: str | None
    model_stage:   str | None
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Structured error response — never return raw tracebacks to clients."""
    error:   str
    detail:  str | None = None
    code:    int


# ── Batch prediction models ─────────────────────────────────────────────────────

class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request — up to 500 customers in a single call.
    Predictions are run in one vectorized forward pass.
    """
    requests: list[ChurnFeatures] = Field(
        min_length=1,
        description="List of customer feature records (1 – BATCH_MAX_SIZE).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "requests": [
                    {
                        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                        "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
                        "MultipleLines": "No", "InternetService": "Fiber optic",
                        "OnlineSecurity": "No", "OnlineBackup": "Yes",
                        "DeviceProtection": "No", "TechSupport": "No",
                        "StreamingTV": "Yes", "StreamingMovies": "No",
                        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                        "PaymentMethod": "Electronic check",
                        "MonthlyCharges": 75.35, "TotalCharges": 904.20,
                    }
                ]
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """Batch prediction result."""
    predictions: list[PredictionResponse] = Field(description="Per-request results in input order.")
    count:        int                       = Field(description="Number of predictions returned.")
    model_version: str                      = Field(description="Model version used for all predictions.")

