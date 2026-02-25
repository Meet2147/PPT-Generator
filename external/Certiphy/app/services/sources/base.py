from abc import ABC, abstractmethod
from typing import List, Dict


class BlueprintSource(ABC):
    """
    Base class for all certification blueprint / study guide fetchers.

    Each source (Microsoft Learn, AWS exam guides, Google Cloud, etc.)
    must return a list of sections in the format:

    [
        {
            "title": "Section title",
            "body": "Full textual content for this section"
        },
        ...
    ]
    """

    @abstractmethod
    async def fetch_sections(self, blueprint_url: str) -> List[Dict[str, str]]:
        """
        Fetch and parse official exam blueprint/study guide content.

        Args:
            blueprint_url (str): URL of the official exam guide.

        Returns:
            List[Dict[str, str]]: Parsed sections with title + body.
        """
        raise NotImplementedError