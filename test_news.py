import asyncio
from src.data.news_fetcher import CryptoNewsFetcher

async def main():
    fetcher = CryptoNewsFetcher()
    print("Fetching BTC news from CryptoCompare...")
    headlines = await fetcher.fetch_news("BTC")
    
    if not headlines:
        print("No headlines fetched. Please check logs.")
        return
        
    print(f"\n--- Successfully fetched {len(headlines)} headlines ---")
    for i, headline in enumerate(headlines, 1):
        print(f"{i}. {headline}")

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
