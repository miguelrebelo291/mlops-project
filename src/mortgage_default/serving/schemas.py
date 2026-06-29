from typing import Any

from pydantic import BaseModel, Field


class LoanRequest(BaseModel):
    """One raw loan/origination record."""

    features: dict[str, Any]
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class BatchLoanRequest(BaseModel):
    """Many raw loan/origination records."""

    rows: list[dict[str, Any]] = Field(..., min_length=1)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)