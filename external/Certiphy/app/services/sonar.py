import os
import httpx
import json

def _env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v

async def sonar_generate_json(prompt: str, schema: dict, timeout_s: float = 60.0) -> dict:
    """
    Forces the model to return JSON matching the provided JSON Schema.
    """
    api_key = _env("SONAR_API_KEY")
    base_url = os.getenv("SONAR_BASE_URL", "https://api.perplexity.ai").rstrip("/")

    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Return only structured JSON that matches the schema."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": schema},
        },
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )

    # Helpful error message if auth/endpoint issues happen
    if resp.status_code >= 400:
        raise RuntimeError(f"Sonar error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

    if not content:
        raise RuntimeError(f"Sonar returned empty content. Full response: {json.dumps(data)[:800]}")

    # With response_format JSON schema, content should be valid JSON string
    return json.loads(content)