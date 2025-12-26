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
        max_tokens: int = 1200,
        # Perplexity-specific (optional)
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[str] = None,   # "day"|"week"|"month"|"year"
        search_mode: Optional[str] = None,             # "web"|"academic"|"sec"
        return_images: bool = False,
        return_related_questions: bool = False,
        # IMPORTANT: new options
        disable_search: bool = True,
        response_format: Optional[Dict[str, Any]] = None,
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
            "disable_search": disable_search,
        }

        # Force structured JSON output whenever we need strict parsing
        if response_format is not None:
            payload["response_format"] = response_format  #  [oai_citation:1‡docs.perplexity.ai](https://docs.perplexity.ai/api-reference/chat-completions-post)

        if search_domain_filter:
            payload["search_domain_filter"] = search_domain_filter
        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter  #  [oai_citation:2‡docs.perplexity.ai](https://docs.perplexity.ai/api-reference/chat-completions-post)
        if search_mode:
            payload["search_mode"] = search_mode  #  [oai_citation:3‡docs.perplexity.ai](https://docs.perplexity.ai/api-reference/chat-completions-post)

        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(url, headers=headers, json=payload)

        if r.status_code != 200:
            raise RuntimeError(f"Perplexity API error {r.status_code}: {r.text[:2000]}")

        data = r.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not isinstance(content, str):
            content = str(content or "")
        return content.strip()
