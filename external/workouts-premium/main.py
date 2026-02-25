import json
import os
import requests
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from typing import List
from pathlib import Path
from perplexity_service import analyze_workout_question
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Exercise Database API",
    description="API to get exercise variations based on muscle groups or body parts",
    version="1.0.0"
)

USAGE_API_BASE_URL = os.getenv("USAGE_API_BASE_URL", "https://getmyworkouts.dashovia.com/api/v1").rstrip("/")
USAGE_API_TIMEOUT_SECONDS = float(os.getenv("USAGE_API_TIMEOUT_SECONDS", "8"))

# Load data from JSON files
def load_json(filename: str):
    """Load JSON data from file"""
    file_path = Path(__file__).parent / filename
    with open(file_path, 'r') as f:
        return json.load(f)

# Load all data on startup
exercises = load_json("exercises.json")
muscles = load_json("muscles.json")
body_parts = load_json("bodyParts.json")

# Mount static file directories for GIFs
static_dir = Path(__file__).parent
GIF_DEFAULT_SIZE_DIR = static_dir / "gifs_360x360"
GIF_PROXY_ROUTE = "/media/gif"

# Extract muscle and body part names for quick lookup
muscle_names = {m["name"].lower(): m["name"] for m in muscles}
body_part_names = {b["name"].lower(): b["name"] for b in body_parts}
target_muscle_set = {
    tm.lower()
    for e in exercises
    for tm in e.get("targetMuscles", [])
}

# Map common user/LLM terms to exercise dataset targetMuscles values.
target_muscle_aliases = {
    "deltoid": "delts",
    "deltoids": "delts",
    "shoulder": "delts",
    "shoulders": "delts",
    "chest": "pectorals",
    "pec": "pectorals",
    "pecs": "pectorals",
    "back": "upper back",
}


def normalize_target_muscle(value: str) -> str:
    normalized = value.lower().strip()
    if normalized in target_muscle_set:
        return normalized
    return target_muscle_aliases.get(normalized, normalized)


def get_gif_url(request: Request, gif_filename: str) -> str:
    """Return internal API route for GIFs so origin URLs are not exposed."""
    return f"{GIF_PROXY_ROUTE}/{gif_filename}"


def resolve_api_key(request: Request) -> str:
    header_key = request.headers.get("x-api-key", "").strip()
    if header_key:
        return header_key

    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    return ""


def _usage_api_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return str(payload.get("detail") or payload.get("message") or payload)
        return str(payload)
    except Exception:
        return response.text or "Usage service request failed."


def validate_api_key_with_usage_service(api_key: str) -> None:
    try:
        response = requests.get(
            f"{USAGE_API_BASE_URL}/usage",
            headers={"x-api-key": api_key, "accept": "application/json"},
            timeout=USAGE_API_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Unable to reach usage service for API key validation.",
        )

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Usage service validation failed: {_usage_api_error_detail(response)}",
        )


def consume_usage_with_service(api_key: str, usage_type: str) -> None:
    if usage_type == "standard":
        endpoint = "consume-standard"
    elif usage_type == "ai":
        endpoint = "consume-ai"
    else:
        raise HTTPException(status_code=500, detail="Invalid usage type configuration.")

    try:
        response = requests.post(
            f"{USAGE_API_BASE_URL}/{endpoint}",
            headers={"x-api-key": api_key, "accept": "application/json"},
            timeout=USAGE_API_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="Unable to reach usage service.")

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")

    if response.status_code == 429:
        raise HTTPException(status_code=429, detail=_usage_api_error_detail(response))

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Usage service consume failed: {_usage_api_error_detail(response)}",
        )


def require_api_key(request: Request) -> str:
    api_key = resolve_api_key(request)
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key.")
    validate_api_key_with_usage_service(api_key)
    return api_key


@app.get(f"{GIF_PROXY_ROUTE}" + "/{gif_filename}")
async def get_gif(gif_filename: str, api_key: str = Depends(require_api_key)):
    """Serve 360x360 GIFs through an API route."""
    safe_name = Path(gif_filename).name
    if safe_name != gif_filename or not safe_name.lower().endswith(".gif"):
        raise HTTPException(status_code=400, detail="Invalid GIF filename")

    gif_path = GIF_DEFAULT_SIZE_DIR / safe_name
    if not gif_path.exists():
        raise HTTPException(status_code=404, detail="GIF not found")

    consume_usage_with_service(api_key, "standard")
    return FileResponse(path=gif_path, media_type="image/gif")


def format_exercise(exercise: dict, request: Request = None, include_instructions: bool = False) -> dict:
    """Format exercise data with full GIF URL"""
    formatted = {
        "exerciseId": exercise["exerciseId"],
        "name": exercise["name"],
        "gifUrl": exercise["gifUrl"],
        "equipment": exercise.get("equipments", []),
        "secondaryMuscles": exercise.get("secondaryMuscles", [])
    }
    
    # Add full GIF URL if request is provided
    if request:
        formatted["gifUrl"] = get_gif_url(request, exercise["gifUrl"])
    
    if include_instructions:
        formatted["instructions"] = exercise.get("instructions", [])
        formatted["targetMuscles"] = exercise.get("targetMuscles", [])
        formatted["bodyParts"] = exercise.get("bodyParts", [])
    
    return formatted


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Exercise Database API",
        "endpoints": {
            "/available-muscles": "Get all available muscles",
            "/available-body-parts": "Get all available body parts",
            "/exercises-by-muscles": "Get exercises by muscle groups (select 2)",
            "/exercises-by-body-parts": "Get exercises by body parts (select 2)"
        }
    }


@app.get("/available-muscles", response_model=List[str])
async def get_available_muscles(api_key: str = Depends(require_api_key)):
    """Get all available muscle groups"""
    result = sorted([m["name"] for m in muscles])
    consume_usage_with_service(api_key, "standard")
    return result


@app.get("/available-body-parts", response_model=List[str])
async def get_available_body_parts(api_key: str = Depends(require_api_key)):
    """Get all available body parts"""
    result = sorted([b["name"] for b in body_parts])
    consume_usage_with_service(api_key, "standard")
    return result


@app.get("/exercises-by-muscles")
async def get_exercises_by_muscles(
    muscle1: str = Query(..., description="First muscle group"),
    muscle2: str = Query(..., description="Second muscle group"),
    request: Request = None,
    api_key: str = Depends(require_api_key),
):
    """
    Get 7-10 exercise variations for each of 2 selected muscle groups.
    
    Example: /exercises-by-muscles?muscle1=biceps&muscle2=triceps
    """
    # Normalize input
    muscle1_normalized = muscle1.lower().strip()
    muscle2_normalized = muscle2.lower().strip()
    
    # Validate muscles exist
    if muscle1_normalized not in muscle_names or muscle2_normalized not in muscle_names:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid muscle. Available muscles: {', '.join(sorted(muscle_names.keys()))}"
        )
    
    # Get original case names
    muscle1_original = muscle_names[muscle1_normalized]
    muscle2_original = muscle_names[muscle2_normalized]
    
    # Filter exercises by target muscles
    exercises_muscle1 = [
        {
            "exerciseId": e["exerciseId"],
            "name": e["name"],
            "gifUrl": get_gif_url(request, e["gifUrl"]),
            "equipment": e.get("equipments", []),
            "bodyPart": e.get("bodyParts", [])[0] if e.get("bodyParts") else None,
            "secondaryMuscles": e.get("secondaryMuscles", [])
        }
        for e in exercises
        if muscle1_original in e.get("targetMuscles", [])
    ][:10]  # Limit to 10
    
    exercises_muscle2 = [
        {
            "exerciseId": e["exerciseId"],
            "name": e["name"],
            "gifUrl": get_gif_url(request, e["gifUrl"]),
            "equipment": e.get("equipments", []),
            "bodyPart": e.get("bodyParts", [])[0] if e.get("bodyParts") else None,
            "secondaryMuscles": e.get("secondaryMuscles", [])
        }
        for e in exercises
        if muscle2_original in e.get("targetMuscles", [])
    ][:10]  # Limit to 10
    
    result = {
        "selectedMuscles": [muscle1_original, muscle2_original],
        "results": {
            muscle1_original: {
                "count": len(exercises_muscle1),
                "exercises": exercises_muscle1
            },
            muscle2_original: {
                "count": len(exercises_muscle2),
                "exercises": exercises_muscle2
            }
        }
    }
    consume_usage_with_service(api_key, "standard")
    return result


@app.get("/exercises-by-body-parts")
async def get_exercises_by_body_parts(
    part1: str = Query(..., description="First body part"),
    part2: str = Query(..., description="Second body part"),
    request: Request = None,
    api_key: str = Depends(require_api_key),
):
    """
    Get 7-10 exercise variations for each of 2 selected body parts.
    
    Example: /exercises-by-body-parts?part1=chest&part2=back
    """
    # Normalize input
    part1_normalized = part1.lower().strip()
    part2_normalized = part2.lower().strip()
    
    # Validate body parts exist
    if part1_normalized not in body_part_names or part2_normalized not in body_part_names:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid body part. Available body parts: {', '.join(sorted(body_part_names.keys()))}"
        )
    
    # Get original case names
    part1_original = body_part_names[part1_normalized]
    part2_original = body_part_names[part2_normalized]
    
    # Filter exercises by body parts
    exercises_part1 = [
        {
            "exerciseId": e["exerciseId"],
            "name": e["name"],
            "gifUrl": get_gif_url(request, e["gifUrl"]),
            "equipment": e.get("equipments", []),
            "targetMuscles": e.get("targetMuscles", []),
            "secondaryMuscles": e.get("secondaryMuscles", [])
        }
        for e in exercises
        if part1_original in e.get("bodyParts", [])
    ][:10]  # Limit to 10
    
    exercises_part2 = [
        {
            "exerciseId": e["exerciseId"],
            "name": e["name"],
            "gifUrl": get_gif_url(request, e["gifUrl"]),
            "equipment": e.get("equipments", []),
            "targetMuscles": e.get("targetMuscles", []),
            "secondaryMuscles": e.get("secondaryMuscles", [])
        }
        for e in exercises
        if part2_original in e.get("bodyParts", [])
    ][:10]  # Limit to 10
    
    result = {
        "selectedBodyParts": [part1_original, part2_original],
        "results": {
            part1_original: {
                "count": len(exercises_part1),
                "exercises": exercises_part1
            },
            part2_original: {
                "count": len(exercises_part2),
                "exercises": exercises_part2
            }
        }
    }
    consume_usage_with_service(api_key, "standard")
    return result


@app.get("/exercise/{exercise_id}")
async def get_exercise_details(
    exercise_id: str,
    request: Request = None,
    api_key: str = Depends(require_api_key),
):
    """Get detailed information about a specific exercise"""
    exercise = next((e for e in exercises if e["exerciseId"] == exercise_id), None)
    
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")
    
    # Format the exercise with full GIF URL
    result = {
        "exerciseId": exercise["exerciseId"],
        "name": exercise["name"],
        "gifUrl": get_gif_url(request, exercise["gifUrl"]),
        "targetMuscles": exercise.get("targetMuscles", []),
        "bodyParts": exercise.get("bodyParts", []),
        "equipments": exercise.get("equipments", []),
        "secondaryMuscles": exercise.get("secondaryMuscles", []),
        "instructions": exercise.get("instructions", [])
    }
    
    consume_usage_with_service(api_key, "standard")
    return result


@app.post("/suggest-workouts")
async def suggest_workouts(
    question: str = Query(..., description="User's workout question"),
    request: Request = None,
    api_key: str = Depends(require_api_key),
):
    """
    AI-powered endpoint that suggests workouts based on user questions using Perplexity Sonar.
    
    Example: /suggest-workouts?question=I want to build bigger arms and shoulder muscles
    """
    try:
        # Get available options
        muscle_list = [m["name"] for m in muscles]
        body_part_list = [b["name"] for b in body_parts]
        
        # Analyze question with Perplexity
        analysis = analyze_workout_question(question, muscle_list, body_part_list)
        
        # Get exercises based on suggested muscles (with normalization)
        exercises_list = []
        matched_any = False
        
        if analysis.get("suggested_muscles"):
            for muscle in analysis["suggested_muscles"]:
                normalized_muscle = normalize_target_muscle(muscle)
                exercises_by_muscle = [
                    {
                        "exerciseId": e["exerciseId"],
                        "name": e["name"],
                        "gifUrl": get_gif_url(request, e["gifUrl"]),
                        "equipment": e.get("equipments", []),
                        "targetMuscles": e.get("targetMuscles", []),
                        "secondaryMuscles": e.get("secondaryMuscles", []),
                        "bodyParts": e.get("bodyParts", [])
                    }
                    for e in exercises
                    if normalized_muscle in [tm.lower() for tm in e.get("targetMuscles", [])]
                ][:8]  # Limit to 8 per muscle
                if exercises_by_muscle:
                    matched_any = True
                exercises_list.extend(exercises_by_muscle)

        # Fallback: if no muscle match, try body-part based matching.
        if not matched_any and analysis.get("suggested_body_parts"):
            for body_part in analysis["suggested_body_parts"]:
                normalized_part = body_part.lower().strip()
                exercises_by_part = [
                    {
                        "exerciseId": e["exerciseId"],
                        "name": e["name"],
                        "gifUrl": get_gif_url(request, e["gifUrl"]),
                        "equipment": e.get("equipments", []),
                        "targetMuscles": e.get("targetMuscles", []),
                        "secondaryMuscles": e.get("secondaryMuscles", []),
                        "bodyParts": e.get("bodyParts", [])
                    }
                    for e in exercises
                    if normalized_part in [bp.lower() for bp in e.get("bodyParts", [])]
                ][:8]
                exercises_list.extend(exercises_by_part)
        
        # Remove duplicates based on exerciseId
        seen = set()
        unique_exercises = []
        for ex in exercises_list:
            if ex["exerciseId"] not in seen:
                seen.add(ex["exerciseId"])
                unique_exercises.append(ex)
        
        result = {
            "userQuestion": question,
            "analysis": analysis,
            "suggestedWorkouts": unique_exercises,
            "totalSuggestions": len(unique_exercises)
        }
        consume_usage_with_service(api_key, "ai")
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
