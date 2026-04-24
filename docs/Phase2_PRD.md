# Phase 2: The Macro Oracle

## Overview
The Macro Oracle is the second autonomous agent in the Apex Intelligence Engine. While the Quant Agent focuses on high-frequency microstructure and order book dynamics, the Macro Oracle is designed to process unstructured text data to gauge market sentiment and identify macroeconomic catalysts.

## Epic 5: Data Ingestion & Filtration
- **Task 5.1:** Integrate News APIs (e.g., CryptoPanic, NewsAPI, Twitter API) to stream real-time financial headlines and tweets.
- **Task 5.2:** Build an Information Filtration pipeline to discard noise and isolate high-impact news related to our traded assets (e.g., BTC, ETH).

## Epic 6: Sentiment Engine (NLP)
- **Task 6.1:** Implement the `SentimentAnalyzer` to score headlines.
- **Task 6.2:** Integrate a pre-trained financial NLP model (e.g., FinBERT) for robust evaluation.
- **Task 6.3:** Develop aggregation logic to compute a rolling "Macro Sentiment Score" across various time windows.
- **Task 6.4:** Route the computed sentiment scores to TimescaleDB for historical analysis and model training.

## Epic 7: Oracle Integration
- **Task 7.1:** Expose the Macro Sentiment Score via a FastAPI endpoint.
- **Task 7.2:** Update the `dataset_builder.py` in Phase 1 to fetch the latest Macro Sentiment Score as a new feature for the Quant Model.
