"""Pydantic request/response models for the prediction API."""

from pydantic import BaseModel, Field, field_validator

MAX_URL_LENGTH = 2048  # generous real-world cap (browsers/servers typically cap ~2000)
MIN_URL_LENGTH = 4     # matches the Phase 1 cleaning threshold for a plausible URL


class PredictRequest(BaseModel):
    url: str = Field(..., description="The URL to classify.", examples=["https://www.google.com"])

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("url must not be empty or whitespace-only")
        if len(v) < MIN_URL_LENGTH:
            raise ValueError(f"url must be at least {MIN_URL_LENGTH} characters")
        if len(v) > MAX_URL_LENGTH:
            raise ValueError(f"url must be at most {MAX_URL_LENGTH} characters")
        if "\x00" in v:
            raise ValueError("url must not contain null bytes")
        return v


class PredictResponse(BaseModel):
    url: str
    prediction: str = Field(..., description="'phishing' or 'legitimate'")
    phishing_probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float
    model_version: str
    features: dict[str, float | int] = Field(
        default_factory=dict, description="Extracted lexical and structural URL features"
    )
    note: str | None = Field(
        default=None, description="Guardrail or model execution context note"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
