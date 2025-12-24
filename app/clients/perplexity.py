from typing import Any, Dict, Optional
import httpx

PPLX_BASE = "https://api.perplexity.ai"

class PerplexityClient:
    def __init__(self, api_key: str, timeout_s: float = 60.0):
        self.api_key = api_key
        self.timeout = timeout_s

    async def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1800,
        search_recency_filter: Optional[str] = None,  # e.g. "month"
    ) -> str:
        url = f"{PPLX_BASE}/chat/completions"
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
        }

        # Optional knobs (Perplexity supports extra fields; harmless if ignored)
        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        # choices[0].message.content
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )