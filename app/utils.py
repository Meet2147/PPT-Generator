import base64
import json
import re
import json
from typing import Any, Dict, List, Union


def b64encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

class ModelJSONError(Exception):
    pass

def extract_json_object(text: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Robust JSON extractor:
    - Finds first JSON object/array in text
    - Uses json.JSONDecoder().raw_decode for partial + extra text
    - Works even if model returns: "Sure! {...}\n\nMore text"
    """
    if not text or not isinstance(text, str):
        raise ModelJSONError("Empty model output")

    s = text.strip()

    # Find first '{' or '['
    start_candidates = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if not start_candidates:
        excerpt = s[:200].replace("\n", "\\n")
        raise ModelJSONError(f"No valid JSON object/array found in model output; excerpt={excerpt}")

    start = min(start_candidates)
    s2 = s[start:]

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(s2)
        return obj
    except json.JSONDecodeError:
        # Try progressive trimming from end (handles truncated tail text)
        for cut in range(len(s2), max(len(s2) - 4000, 0), -1):
            chunk = s2[:cut].rstrip()
            try:
                obj, _ = decoder.raw_decode(chunk)
                return obj
            except Exception:
                continue

        excerpt = s2[:300].replace("\n", "\\n")
        raise ModelJSONError(f"Invalid JSON from model; excerpt={excerpt}")
