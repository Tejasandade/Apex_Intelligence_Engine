import os
import httpx
from loguru import logger
from typing import Dict, List, Any


class IndianMacroOracle:
    """
    Indian Market Macro Oracle.
    Fetches localized domestic financial news and parses FII/DII institutional flow data.
    """

    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        if not self.news_api_key:
            logger.warning("NEWS_API_KEY not found in environment. fetch_domestic_news will fail.")

    async def fetch_domestic_news(self) -> List[Dict[str, Any]]:
        """
        Pulls the top 20 Indian financial/business headlines using NewsAPI.
        """
        if not self.news_api_key:
            return []

        url = f"https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey={self.news_api_key}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                articles = data.get("articles", [])
                logger.info(f"Fetched {len(articles)} domestic news headlines from NewsAPI.")
                return articles
        except Exception as exc:
            logger.error(f"Failed to fetch domestic news: {exc}")
            return []

    async def fetch_fii_dii_data(self) -> Dict[str, Any]:
        """
        Returns a mock dictionary of simulated Foreign and Domestic Institutional
        net buying/selling activity.
        """
        logger.debug("Generating simulated FII/DII institutional flow data.")
        return {
            "FII_net_crores": 1250.50,
            "DII_net_crores": -400.25,
            "net_institutional_flow_crores": 850.25,
            "sentiment": "Bullish",
        }
