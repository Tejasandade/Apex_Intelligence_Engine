import asyncio
from loguru import logger
from transformers import pipeline
from src.data.news_fetcher import CryptoNewsFetcher
from src.data.news_filter import NewsFilter
from src.data.db import db_manager

class SentimentAnalyzer:
    """
    Core component of the Macro Oracle.
    Responsible for evaluating unstructured text (news headlines, tweets)
    and generating quantitative sentiment scores.
    """
    
    def __init__(self):
        logger.info("Initializing SentimentAnalyzer with ProsusAI/finbert model...")
        self.nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        self.is_ready = True
        
    def analyze_headline(self, text: str) -> dict:
        """
        Evaluates a single headline using FinBERT and returns a sentiment score.
        """
        # Run inference using FinBERT
        result = self.nlp(text)[0]
        
        label = result['label'].lower()
        confidence = float(result['score'])
        
        # Map labels to numerical sentiment score
        if label == 'positive':
            score = 0.8
        elif label == 'negative':
            score = 0.2
        else:
            score = 0.5
            
        return {
            "text": text,
            "sentiment_score": score,
            "confidence": confidence,
            "raw_label": label
        }

class MacroOracle:
    def __init__(self):
        self.fetcher = CryptoNewsFetcher()
        self.filter = NewsFilter()
        self.analyzer = SentimentAnalyzer()
        
    async def run_pipeline(self):
        logger.info("Starting Macro Oracle NLP Pipeline...")
        
        # 1. Fetch News
        headlines = await self.fetcher.fetch_news("BTC")
        if not headlines:
            logger.warning("No headlines fetched. Aborting pipeline.")
            return
            
        logger.info(f"Fetched {len(headlines)} raw headlines.")
        
        # 2. Filter by Asset
        asset_keywords = ["BTC", "Bitcoin"]
        asset_filtered = self.filter.filter_by_asset(headlines, asset_keywords)
        
        # 3. Remove Noise
        clean_headlines = self.filter.remove_noise(asset_filtered)
        logger.info(f"Filtration complete. {len(clean_headlines)} high-value headlines retained.")
        
        # 4. Analyze Sentiment
        scores = []
        for headline in clean_headlines:
            result = self.analyzer.analyze_headline(headline)
            scores.append(result['sentiment_score'])
            logger.debug(f"Sentiment [Score: {result['sentiment_score']}]: {result['text']}")
            
        # 5. Aggregate and Save Score
        average_score = sum(scores) / len(scores) if scores else 0.5
        logger.info(f"Calculated Rolling Macro Sentiment Score: {average_score:.4f}")
        
        await db_manager.insert_sentiment(average_score)

async def main():
    await db_manager.connect()
    try:
        oracle = MacroOracle()
        await oracle.run_pipeline()
    finally:
        await db_manager.disconnect()

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())
