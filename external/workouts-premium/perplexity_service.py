import os
import json
import re
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_BASE_URL = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
# Perplexity exposes an OpenAI-compatible endpoint for chat completions
PERPLEXITY_CHAT_COMPLETIONS_URL = f"{PERPLEXITY_BASE_URL.rstrip('/')}/chat/completions"


class PerplexityError(RuntimeError):
    pass


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Best-effort extraction of the first JSON object from a string."""
    text = text.strip()

    # Common case: the response is already pure JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Best-effort: find the first {...} block (non-greedy)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise json.JSONDecodeError("No JSON object found in response", text, 0)

    return json.loads(match.group(0))


def analyze_workout_question(
    user_question: str,
    available_muscles: List[str],
    available_body_parts: List[str],
) -> Dict[str, Any]:
    """Use Perplexity Sonar to analyze a workout question and extract workout preferences.

    Returns:
    {
        "intent": "description of what user wants",
        "suggested_muscles": ["muscle1", "muscle2"],
        "suggested_body_parts": ["part1", "part2"],
        "fitness_level": "beginner/intermediate/advanced",
        "notes": "any special considerations"
    }
    """

    if not PERPLEXITY_API_KEY:
        raise PerplexityError(
            "Missing PERPLEXITY_API_KEY. Add it to your environment or .env file."
        )

    prompt = f"""Analyze this workout question and extract the user's fitness intent.

User Question: \"{user_question}\"

Available Muscles: {', '.join(available_muscles)}
Available Body Parts: {', '.join(available_body_parts)}

Response in JSON format ONLY (no markdown, no extra text):
{{
  \"intent\": \"what the user wants to accomplish\",
  \"suggested_muscles\": [\"pick up to 2 from available list\"],
  \"suggested_body_parts\": [\"pick up to 2 from available list\"],
  \"fitness_level\": \"beginner/intermediate/advanced\",
  \"notes\": \"any special considerations or modifications\"
}}

Return only valid JSON."""

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": "sonar",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        # Keep output deterministic and short
        "temperature": 0.2,
    }

    try:
        resp = requests.post(
            PERPLEXITY_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=45,
        )

        if resp.status_code >= 400:
            # Try to include useful error info
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise PerplexityError(f"Perplexity API error {resp.status_code}: {err}")

        data = resp.json()

        # OpenAI-compatible shape: choices[0].message.content
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not content:
            raise PerplexityError(f"Unexpected Perplexity response shape: {data}")

        result = _extract_json_object(content)
        return result

    except json.JSONDecodeError:
        # Fallback if parsing fails
        return {
            "intent": "general workout",
            "suggested_muscles": available_muscles[:2],
            "suggested_body_parts": available_body_parts[:2],
            "fitness_level": "intermediate",
            "notes": "Perplexity analysis completed but returned unexpected format",
        }
    except requests.RequestException as e:
        raise PerplexityError(f"Network error calling Perplexity: {str(e)}")
    except PerplexityError:
        raise
    except Exception as e:
        raise PerplexityError(f"Unexpected error: {str(e)}")
