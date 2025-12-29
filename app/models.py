
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

InputMode = Literal["text", "image"]

# ---------- Identify ----------

class FoodCandidate(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    normalized_name: str
    cuisine: Optional[str] = None
    is_packaged: Optional[bool] = None
    notes: Optional[str] = None


class IdentifyRequest(BaseModel):
    mode: InputMode = "text"
    text: Optional[str] = None
    hints: Optional[List[str]] = None


class IdentifyResponse(BaseModel):
    candidates: List[FoodCandidate]
    chosen: FoodCandidate


# ---------- Portion ----------

class PortionRequest(BaseModel):
    food_name: str
    mode: InputMode = "text"
    text_context: Optional[str] = None
    assumed_servings: float = Field(default=1.0, ge=0.1)
    household_measure: Optional[str] = None


class PortionEstimate(BaseModel):
    servings: float = Field(ge=0.0)
    grams_total: float = Field(ge=0.0)
    items_count: Optional[float] = None
    household: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    assumptions: List[str] = Field(default_factory=list)


class PortionResponse(BaseModel):
    food_name: str
    portion: PortionEstimate


# ---------- Nutrition ----------

class NutrientsRequest(BaseModel):
    food_name: str
    portion: PortionEstimate
    region: Optional[str] = "IN"
    brand: Optional[str] = None
    include_per_100g: bool = True


class NutrientItem(BaseModel):
    name: str
    amount: float
    unit: str
    per_100g_amount: Optional[float] = None
    daily_value_percent: Optional[float] = None


class NutrientsResponse(BaseModel):
    food_name: str
    portion: PortionEstimate
    calories_kcal: float

    macros: Dict[str, NutrientItem]
    micros: Dict[str, NutrientItem]
    vitamins: Dict[str, NutrientItem]
    minerals: Dict[str, NutrientItem]

    ingredients_guess: List[str] = Field(default_factory=list)
    allergens_guess: List[str] = Field(default_factory=list)
    data_sources: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# ---------- Aggregate ----------


class AnalyzeItem(BaseModel):
    food_name: str
    portion: "PortionResponse"
    nutrients: "NutrientsResponse"


class AnalyzeImageNutritionItem(BaseModel):
    name: str

    calories: float = 0.0
    serving_size_g: float = Field(default=0.0, ge=0)

    fat_total_g: float = 0.0
    fat_saturated_g: float = 0.0
    protein_g: float = 0.0

    sodium_mg: float = 0.0
    potassium_mg: float = 0.0
    cholesterol_mg: float = 0.0

    carbohydrates_total_g: float = 0.0
    fiber_g: float = 0.0
    sugar_g: float = 0.0




class AnalyzeImageRequest(BaseModel):
    hints: Optional[List[str]] = None
    region: str = "IN"
    
class AnalyzeResponse(BaseModel):
    identify: "IdentifyResponse"
    portion: "PortionResponse"
    nutrients: "NutrientsResponse"
    cost_tier: dict
    items: Optional[List[AnalyzeItem]] = None


    # ---------- PPT Generator ----------

class PPTGenerateRequest(BaseModel):
    topic: str
    design_number: int = Field(default=2, ge=1, le=7)

class PPTGenerateResponse(BaseModel):
    status: str
    filename: str
    # download_url: str
    fallback_used: bool = False
