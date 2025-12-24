import base64
import json
import re
import json
from typing import Any, Dict, List, Union

class ModelJSONError(Exception):
    pass



def b64encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

class ModelJSONError(Exception):
    pass

def _extract_first_json_block(s: str) -> str:
    """
    Returns the first valid JSON object/array block from a string by scanning braces/brackets.
    Handles nested JSON and braces inside quoted strings.
    """
    if not s or not isinstance(s, str):
        raise ModelJSONError("Empty model output")

    start = None
    stack: List[str] = []
    in_string = False
    escape = False

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue

        if ch in "{[":
            if start is None:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            opener = stack.pop()
            if (opener == "{" and ch != "}") or (opener == "[" and ch != "]"):
                # mismatched, reset and continue searching
                start = None
                stack = []
                continue

            if start is not None and not stack:
                return s[start : i + 1]

    raise ModelJSONError("No valid JSON object/array found in model output")

def extract_json_object(text: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Extracts and parses the first JSON object/array found in 'text'.
    Raises ModelJSONError if parsing fails.
    """
    block = _extract_first_json_block(text.strip())

    try:
        return json.loads(block)
    except Exception as e:
        excerpt = block[:300].replace("\n", "\\n")
        raise ModelJSONError(f"Invalid JSON from model: {e}; excerpt={excerpt}")