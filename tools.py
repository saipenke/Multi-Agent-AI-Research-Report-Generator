"""Custom CrewAI tool wrapping DuckDuckGo search (no API key needed)."""
import time
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS


class SearchInput(BaseModel):
    query: str = Field(..., description="The web search query to run.")


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Searches the web with DuckDuckGo and returns titles, URLs and snippets. "
        "Use specific queries. Run several different queries to cover a topic."
    )
    args_schema: Type[BaseModel] = SearchInput
    max_results: int = 4       # fewer results = fewer tokens
    snippet_chars: int = 200   # shorter snippets = fewer tokens

    def _run(self, query: str) -> str:
        last_error = None
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=self.max_results))
                if not results:
                    return f"No results found for: {query}"

                lines = []
                for i, r in enumerate(results, 1):
                    title = r.get("title", "No title")
                    url = r.get("href") or r.get("url", "")
                    body = (r.get("body") or "")[: self.snippet_chars]
                    lines.append(f"[{i}] {title}\nURL: {url}\nSummary: {body}")
                return "\n\n".join(lines)
            except Exception as e:  # rate limits etc.
                last_error = e
                time.sleep(2 * (attempt + 1))

        return f"Search failed after retries: {last_error}"