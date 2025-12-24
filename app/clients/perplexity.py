# app/clients/perplexity.py
from __future__ import annotations

from typing import Optional
import httpx

class PerplexityClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"

    async def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("Missing PPLX_API_KEY / pplx_api_key")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)

        # If 400: return full detail for debugging in your API response/logs
        if r.status_code != 200:
            raise RuntimeError(f"Perplexity API error {r.status_code}: {r.text}")

        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return f"""
            {{
              "error": "perplexity_error",
              "status": {r.status_code},
              "message": {r.text!r}
            }}
            """
