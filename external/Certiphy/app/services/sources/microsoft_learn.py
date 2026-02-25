import httpx
from bs4 import BeautifulSoup
from typing import List, Dict
from .base import BlueprintSource

class MicrosoftLearnSource(BlueprintSource):
    async def fetch_sections(self, blueprint_url: str) -> List[Dict[str, str]]:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CertQuiz/1.0)"},
        ) as client:
            r = await client.get(blueprint_url)
            r.raise_for_status()

        soup = BeautifulSoup(r.text, "lxml")
        main = soup.find("main") or soup

        h2s = main.find_all("h2")
        sections: List[Dict[str, str]] = []

        for h2 in h2s:
            title = h2.get_text(" ", strip=True)
            if not title:
                continue

            chunks = []
            for sib in h2.find_all_next():
                if sib.name == "h2":
                    break
                if sib.name in ("p", "li"):
                    txt = sib.get_text(" ", strip=True)
                    if txt:
                        chunks.append(txt)
                if sum(len(c) for c in chunks) > 2500:
                    break

            body = "\n".join(chunks).strip()
            if body:
                sections.append({"title": title, "body": body})

        if not sections:
            text = main.get_text("\n", strip=True)
            sections = [{"title": "Study Guide", "body": text[:4000]}]

        return sections