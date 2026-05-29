import asyncio
from src.models.training.trainer import UnifiedTrainer
from src.core.logging import get_logger

logger = get_logger("apex.train_all")

async def main():
    logger.info("Starting model training for btcusdt")
    trainer = UnifiedTrainer(market_type="crypto", symbol="btcusdt")
    results = await trainer.run(save_models=True)
    logger.info("Training complete", results=results)

if __name__ == "__main__":
    asyncio.run(main())
