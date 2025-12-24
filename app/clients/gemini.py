import base64
from typing import Optional
import httpx


class GeminiClient:
    def __init__(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        if not api_key:
            raise ValueError("Gemini API key is missing")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def generate_text(
        self,
        model: str,
        system: str,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> str:
        url = f"{self.base_url}/models/{model}:generateContent"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system}\n\n{prompt}"}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, params={"key": self.api_key}, json=payload)
            # helpful debug on 400
            if r.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"{r.status_code} {r.reason_phrase}: {r.text}",
                    request=r.request,
                    response=r,
                )

        data = r.json()
        return self._extract_text(data)

    async def generate_with_image(
        self,
        model: str,
        system: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> str:
        if not image_bytes:
            raise ValueError("image_bytes is empty")

        url = f"{self.base_url}/models/{model}:generateContent"
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{system}\n\n{prompt}"},
                        {"inlineData": {"mimeType": mime_type, "data": b64}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, params={"key": self.api_key}, json=payload)
            if r.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"{r.status_code} {r.reason_phrase}: {r.text}",
                    request=r.request,
                    response=r,
                )

        data = r.json()
        return self._extract_text(data)

    def _extract_text(self, data: dict) -> str:
        """
        Expected response shape:
        { "candidates": [ { "content": { "parts": [ {"text": "..."} ] } } ] }
        """
        try:
            candidates = data.get("candidates") or []
            if not candidates:
                return ""
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
            return "".join(texts).strip()
        except Exception:
            return ""