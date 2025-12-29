# app/clients/perplexity.py
from __future__ import annotations

from typing import Any, Dict, Optional, List
import json
import httpx


class PerplexityClient:
    def __init__(self, api_key: str, base_url: str = "https://api.perplexity.ai"):
        if not api_key:
            raise ValueError("pplx_api_key is missing")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def chat_text(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 900,
        # optional search knobs (only include if you KNOW your plan/model supports them)
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[str] = None,
        search_mode: Optional[str] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "text"},
        }

        # Only include when provided AND supported (otherwise Perplexity can 400)
        if search_domain_filter:
            payload["search_domain_filter"] = search_domain_filter
        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter
        if search_mode:
            payload["search_mode"] = search_mode

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)

        if r.status_code != 200:
            raise RuntimeError(f"Perplexity API error {r.status_code}: {r.text[:2000]}")

        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(content, str):
            content = str(content or "")
        return content.strip()

    async def chat_json_schema(
        self,
        model: str,
        system: str,
        user: str,
        json_schema: Dict[str, Any],
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> Dict[str, Any]:
        """
        Forces the model to return JSON conforming to the schema.
        This avoids your "Invalid JSON from model" issues almost entirely.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema,
            },
        }

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)

        if r.status_code != 200:
            raise RuntimeError(f"Perplexity API error {r.status_code}: {r.text[:2000]}")

        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            content = str(content or "")

        # content should already be JSON, but parse safely:
        return json.loads(content)
