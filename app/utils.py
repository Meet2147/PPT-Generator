# app/utils.py
from __future__ import annotations

from typing import Any
import json


class ModelJSONError(ValueError):
    pass


def extract_json_object(text: str) -> Any:
    """
    Extract the first valid JSON object or array from a string.
    Handles model 'noise' before/after JSON.
    """
    if not text or not isinstance(text, str):
        raise ModelJSONError("Empty model output")

    s = text.strip()

    # Fast path: whole string is JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # Find first '{' or '[' and brace-match
    start_positions = []
    for i, ch in enumerate(s):
        if ch in "{[":
            start_positions.append(i)

    for start in start_positions:
        opening = s[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_str = False
        esc = False

        for j in range(start, len(s)):
            c = s[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue

            if c == '"':
                in_str = True
                continue

            if c == opening:
                depth += 1
            elif c == closing:
                depth -= 1
                if depth == 0:
                    candidate = s[start : j + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break  # try next start position

    excerpt = s[:300].replace("\n", "\\n")
    raise ModelJSONError(f"No valid JSON object/array found in model output. Excerpt={excerpt}")
