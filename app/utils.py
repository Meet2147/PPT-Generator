# app/utils.py
from __future__ import annotations

import json
from typing import Any, Optional


class ModelJSONError(Exception):
    pass


def _find_first_json_start(s: str) -> Optional[int]:
    if not s:
        return None
    brace = s.find("{")
    brack = s.find("[")
    if brace == -1 and brack == -1:
        return None
    if brace == -1:
        return brack
    if brack == -1:
        return brace
    return min(brace, brack)


def extract_json_object(s: str) -> Any:
    """
    Extract the first valid JSON object/array from a text blob.
    Works even if the model adds extra prose before/after.
    Does NOT rely on regex recursion (Python re doesn't support ?R).
    """
    if not isinstance(s, str):
        raise ModelJSONError("Model output is not a string")

    start = _find_first_json_start(s)
    if start is None:
        raise ModelJSONError("No valid JSON object/array found in model output")

    # Scan forward and find the matching closing brace/bracket
    stack = []
    in_string = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        # not in string
        if ch == '"':
            in_string = True
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                # mismatched - abort
                break
            if not stack:
                candidate = s[start : i + 1].strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # try a “looser” repair: cut to last complete char and retry
                    repaired = _loose_repair(candidate)
                    try:
                        return json.loads(repaired)
                    except Exception:
                        excerpt = candidate[:250].replace("\n", "\\n")
                        raise ModelJSONError(f"Invalid JSON from model. Excerpt={excerpt}")

    # If we reached here, JSON was likely truncated
    candidate = s[start:].strip()
    repaired = _truncate_to_balanced(candidate)
    if repaired:
        try:
            return json.loads(repaired)
        except Exception:
            pass

    excerpt = candidate[:250].replace("\n", "\\n")
    raise ModelJSONError(f"Invalid JSON from model (likely truncated). Excerpt={excerpt}")


def _truncate_to_balanced(candidate: str) -> Optional[str]:
    """Try to cut at the last position where braces/brackets are balanced."""
    stack = []
    in_string = False
    escape = False
    last_balanced_end = None

    for i, ch in enumerate(candidate):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                return None
            if not stack:
                last_balanced_end = i

    if last_balanced_end is None:
        return None
    return candidate[: last_balanced_end + 1].strip()


def _loose_repair(s: str) -> str:
    """
    Tiny repair: remove trailing commas before } or ].
    (Does NOT invent missing braces; that's handled by truncation.)
    """
    out = []
    i = 0
    while i < len(s):
        if s[i] == ",":
            j = i + 1
            while j < len(s) and s[j].isspace():
                j += 1
            if j < len(s) and s[j] in "}]":
                i += 1
                continue
        out.append(s[i])
        i += 1
    return "".join(out)
