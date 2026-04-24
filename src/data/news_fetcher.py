import os
import aiohttp
import asyncio
from typing import List
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CryptoNewsFetcher:
    """
    Asynchronous client for fetching real-time crypto news headlines
    from the CryptoCompare (CCData) API.
    """
    def __init__(self):
        self.api_key = os.getenv("CRYPTOCOMPARE_API_KEY")
        self.base_url = "https://min-api.cryptocompare.com/data/v2/news/"
        
        if not self.api_key:
            logger.error("CRYPTOCOMPARE_API_KEY environment variable is missing. News ingestion will fail.")

    async def fetch_news(self, categories: str = "BTC") -> List[str]:
        """
        Fetches the latest news articles for the given categories and extracts their titles.
        Returns a list of clean headline strings.
        """
        if not self.api_key:
            logger.error("Cannot fetch news without an API key.")
            return []

        headers = {
            "authorization": f"Apikey {self.api_key}"
        }
        params = {
            "categories": categories,
            "lang": "EN"
        }
        
        # Increase robustness with explicit timeouts
        timeout = aiohttp.ClientTimeout(total=10)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.base_url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.error(f"CryptoCompare API returned status {response.status}: {await response.text()}")
                        return []
                        
                    data = await response.json()
                    
                    if 'Data' not in data:
                        logger.error("Malformed response from CryptoCompare: 'Data' key missing.")
                        return []
                        
                    # Extract titles from the array of articles
                    titles = [article.get('title', '') for article in data['Data'] if 'title' in article]
                    logger.debug(f"Successfully fetched {len(titles)} headlines for categories: {categories}")
                    
                    return titles

        except asyncio.TimeoutError:
            logger.error(f"Timeout while fetching news for {categories} from CryptoCompare.")
            return []
        except aiohttp.ClientError as e:
            logger.error(f"Network error while fetching news: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in fetch_news: {e}")
            return []
