

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator



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



class ChurnFeatures(BaseModel):

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



class PredictionResponse(BaseModel):
    prediction:       int   = Field(description="0 = No churn, 1 = Churn")
    probability:      float = Field(description="Probability of churn (0.0 – 1.0)")
    model_version:    str   = Field(description="MLflow model version used")
    model_stage:      str   = Field(description="MLflow stage (Production)")
    threshold:        float = Field(description="Decision threshold applied")


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str | None
    model_stage:   str | None
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error:   str
    detail:  str | None = None
    code:    int



class BatchPredictionRequest(BaseModel):
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
    predictions: list[PredictionResponse] = Field(description="Per-request results in input order.")
    count:        int                       = Field(description="Number of predictions returned.")
    model_version: str                      = Field(description="Model version used for all predictions.")
