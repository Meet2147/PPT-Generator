# app/clients/perplexity.py
from __future__ import annotations

from typing import Any, Dict, Optional, List
import httpx


class PerplexityClient:
    def __init__(self, api_key: str, base_url: str = "https://api.perplexity.ai"):
        if not api_key:
            raise ValueError("pplx_api_key is missing")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 900,
        # Perplexity-specific (optional)
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[str] = None,   # "day"|"week"|"month"|"year"
        search_mode: Optional[str] = None,             # "web"|"academic"
        return_images: bool = False,
        return_related_questions: bool = False,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "return_images": return_images,
            "return_related_questions": return_related_questions,
        }

        # only include when provided (don’t send unknown/None)
        if search_domain_filter:
            payload["search_domain_filter"] = search_domain_filter
        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter
        if search_mode:
            payload["search_mode"] = search_mode

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)

        # If Perplexity rejects params/model, this will be non-200
        if r.status_code != 200:
            raise RuntimeError(f"Perplexity API error {r.status_code}: {r.text[:2000]}")

        data = r.json()

        # OpenAI-compatible schema: choices[0].message.content
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not isinstance(content, str):
            content = str(content or "")

        return content.strip()
