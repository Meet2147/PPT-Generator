import json
from typing import Any, Dict, List, Union

class ModelJSONError(Exception):
    pass

def extract_json_object(text: str) -> Union[Dict[str, Any], List[Any]]:
    if not text or not isinstance(text, str):
        raise ModelJSONError("Empty model output")

    s = text.strip()

    # find first JSON start
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if not starts:
        raise ModelJSONError(f"No JSON start found. Excerpt={s[:200]}")

    s = s[min(starts):]
    dec = json.JSONDecoder()

    try:
        obj, _ = dec.raw_decode(s)   # ✅ reads first valid JSON, ignores rest
        return obj
    except json.JSONDecodeError as e:
        raise ModelJSONError(f"Invalid JSON from model. Excerpt={s[:200]}")
