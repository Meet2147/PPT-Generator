from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

# ---------- Common ----------
InputMode = Literal["text", "image"]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="ignore")  # keep API resilient to small model drift


# ---------- Identify ----------
class FoodCandidate(StrictModel):
    name: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    normalized_name: str
    cuisine: Optional[str] = None
    is_packaged: Optional[bool] = None
    notes: Optional[str] = None


class IdentifyRequest(StrictModel):
    mode: InputMode = "text"
    text: Optional[str] = None
    hints: Optional[List[str]] = None


class IdentifyResponse(StrictModel):
    candidates: List[FoodCandidate] = Field(default_factory=list)
    chosen: FoodCandidate


# ---------- Portion ----------
class PortionRequest(StrictModel):
    food_name: str
    mode: InputMode = "text"
    text_context: Optional[str] = None
    assumed_servings: float = Field(default=1.0, ge=0.1)
    household_measure: Optional[str] = None


class PortionEstimate(StrictModel):
    servings: float = Field(default=1.0, ge=0.0)
    grams_total: float = Field(default=0.0, ge=0.0)
    items_count: Optional[float] = Field(default=None, ge=0.0)
    household: Optional[str] = None
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    assumptions: List[str] = Field(default_factory=list)


class PortionResponse(StrictModel):
    food_name: str
    portion: PortionEstimate


# ---------- Nutrition ----------
class NutrientsRequest(StrictModel):
    food_name: str
    portion: PortionEstimate
    region: Optional[str] = "IN"
    brand: Optional[str] = None
    include_per_100g: bool = True


class NutrientItem(StrictModel):
    name: str
    amount: float = 0.0
    unit: str = "g"  # "g" or "mg"
    per_100g_amount: Optional[float] = None
    daily_value_percent: Optional[float] = None


class NutrientsResponse(StrictModel):
    food_name: str
    portion: PortionEstimate
    calories_kcal: float = 0.0

    macros: Dict[str, NutrientItem] = Field(default_factory=dict)
    micros: Dict[str, NutrientItem] = Field(default_factory=dict)
    vitamins: Dict[str, NutrientItem] = Field(default_factory=dict)
    minerals: Dict[str, NutrientItem] = Field(default_factory=dict)

    ingredients_guess: Optional[List[str]] = None
    allergens_guess: Optional[List[str]] = None
    data_sources: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# ---------- Aggregate ----------
class AnalyzeTextRequest(StrictModel):
    text: str
    hints: Optional[List[str]] = None
    region: Optional[str] = "IN"
    include_per_100g: bool = True


# exists for import-compatibility (even if you don’t use it directly in body)
class AnalyzeImageRequest(StrictModel):
    hints: Optional[str] = None          # comma separated in multipart Form
    region: Optional[str] = "IN"
    include_per_100g: bool = True


class AnalyzeResponse(StrictModel):
    identify: IdentifyResponse
    portion: PortionResponse
    nutrients: NutrientsResponse
    cost_tier: Dict[str, str] = Field(default_factory=dict)
