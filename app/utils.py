# app/utils.py
from __future__ import annotations

import json
from typing import Any


class ModelJSONError(Exception):
    pass


def extract_json_object(text: str) -> Any:
    """
    Robustly extract the first JSON object/array from a model response.

    Works even if the model returns:
    - extra text before/after JSON
    - JSON + citations
    - multiple JSON snippets
    - truncated leading whitespace/newlines
    """
    if text is None:
        raise ModelJSONError("No text to parse")

    s = text.strip()
    if not s:
        raise ModelJSONError("Empty model output")

    # Find first '{' or '['
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        raise ModelJSONError("No valid JSON object/array found in model output")

    decoder = json.JSONDecoder()
    tail = s[start:]

    try:
        obj, _idx = decoder.raw_decode(tail)
        return obj
    except Exception:
        # Try again by scanning forward a bit (handles leading junk like ```json)
        for j in range(start + 1, min(len(s), start + 4000)):
            if s[j] in "{[":
                try:
                    obj, _idx = decoder.raw_decode(s[j:])
                    return obj
                except Exception:
                    continue

    excerpt = s[start:start + 250]
    raise ModelJSONError(f"Invalid JSON from model. Excerpt={excerpt}")
