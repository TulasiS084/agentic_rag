"""
Web Search Engine Integration.
Uses DuckDuckGo (ddgs) to fetch real-time and out-of-document information with HTML fallback.
"""
from __future__ import annotations
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Trigger keywords for live/external information
_LIVE_DATA_PATTERNS = [
    r"\b(today|tonight|yesterday|tomorrow|this week|this month|current|currently|now)\b",
    r"\b(stock market|stocks|share price|nasdaq|nifty|sensex|dow jones|sp500|crypto|bitcoin)\b",
    r"\b(news|headline|headlines|breaking|update|updates|latest|recent|recently)\b",
    r"\b(weather|forecast|temperature|rain|climate)\b",
    r"\b(match|game|score|who won|winner|championship|tournament|ipl|fifa)\b",
    r"\b(what happened in|what is happening in|latest version of|release date)\b",
]


def is_live_or_web_query(query: str) -> bool:
    """Detect if a user query requires real-time web search."""
    q_lower = query.lower().strip()
    for pattern in _LIVE_DATA_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for a given query.

    Returns:
        List of dicts: [{"title": str, "url": str, "snippet": str}]
    """
    results: List[Dict[str, str]] = []

    # Attempt 1: ddgs
    try:
        from ddgs import DDGS
        ddgs = DDGS()
        raw_results = list(ddgs.text(query, max_results=max_results))
        for r in raw_results:
            title = r.get("title", "").strip()
            url = r.get("href", "").strip()
            body = r.get("body", "").strip()
            if title and url:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": body,
                })
        if results:
            logger.info(f"ddgs returned {len(results)} results for query '{query}'")
            return results
    except Exception as e:
        logger.warning(f"ddgs search failed: {e}. Trying HTTP fallback.")

    # Attempt 2: Direct DuckDuckGo HTML scraping fallback
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.post(url, data={"q": query}, headers=headers, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            elements = soup.find_all("div", class_="result__body")
            for el in elements[:max_results]:
                title_tag = el.find("a", class_="result__url") or el.find("a", class_="result__snippet")
                snippet_tag = el.find("a", class_="result__snippet")
                if title_tag:
                    href = title_tag.get("href", "")
                    title = title_tag.get_text(strip=True)
                    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                    results.append({
                        "title": title or "Web Source",
                        "url": href,
                        "snippet": snippet,
                    })
            if results:
                return results
    except Exception as e:
        logger.error(f"HTTP fallback web search failed: {e}")

    return results
