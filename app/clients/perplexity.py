import httpx
from typing import Optional

class PerplexityClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"

    async def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        search_recency_filter: Optional[str] = None,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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

            # ✅ FORCE JSON
            "response_format": {"type": "json_object"},
        }

        # optional (keep if you’re using it)
        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter

        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(self.base_url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        content = data["choices"][0]["message"]["content"]
        return content if isinstance(content, str) else str(content)
