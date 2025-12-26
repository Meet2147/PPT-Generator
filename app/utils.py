from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelJSONError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def extract_json_object(text: str) -> Any:
    """
    Robustly extract the FIRST valid JSON object/array from a messy model output.
    Handles extra text before/after JSON. Uses json.JSONDecoder().raw_decode.
    """
    if not isinstance(text, str) or not text.strip():
        raise ModelJSONError("No valid JSON object/array found in model output")

    s = text.strip()

    # Find earliest '{' or '['
    obj_i = s.find("{")
    arr_i = s.find("[")
    starts = [i for i in [obj_i, arr_i] if i != -1]
    if not starts:
        raise ModelJSONError("No valid JSON object/array found in model output")

    start = min(starts)
    s2 = s[start:]

    dec = json.JSONDecoder()
    try:
        parsed, _end = dec.raw_decode(s2)
        return parsed
    except Exception as e:
        excerpt = s2[:220].replace("\n", "\\n")
        raise ModelJSONError(f"Invalid JSON from model. Excerpt={excerpt}") from e
