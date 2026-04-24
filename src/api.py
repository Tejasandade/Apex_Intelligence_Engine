from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from datetime import datetime
from loguru import logger

from src.data.db import db_manager
from src.models.dataset_builder import DataFetcher
from src.models.quant_model import ApexXGBoostModel

# Global model instance
quant_model = ApexXGBoostModel()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Initializing Apex Inference API...")
    await db_manager.connect()
    yield
    # Shutdown logic
    logger.info("Shutting down Apex Inference API...")
    await db_manager.disconnect()

app = FastAPI(
    title="Apex Market Intelligence API",
    description="Inference API for the Apex Quant Model",
    version="2.5",
    lifespan=lifespan
)

@app.get("/inference/{symbol}")
async def get_inference(symbol: str):
    """
    Retrieves the latest market data for the given symbol,
    constructs a feature vector, and returns the directional 
    probability score from the XGBoost Model.
    """
    # Ensure symbol matches the stream format (typically lowercase in our pipeline)
    symbol_formatted = symbol.lower()
    
    try:
        # Build feature dataset
        fetcher = DataFetcher(db_manager)
        df = await fetcher.build_dataset(symbol_formatted)
        
        if df.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"Insufficient data to build feature vector for symbol: {symbol}"
            )
            
        # Run inference
        score = quant_model.predict(df)
        
        # Format response
        return {
            "symbol": symbol.upper(),
            "bullish_probability": round(score, 4),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference pipeline error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during inference")
