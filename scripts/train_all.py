import argparse
import asyncio
from src.models.training.trainer import UnifiedTrainer
from src.core.logging import get_logger

logger = get_logger("apex.train_all")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", type=str, default="crypto")
    parser.add_argument("--symbol", type=str, default="btcusdt")
    args = parser.parse_args()

    logger.info("Starting model training", market=args.market, symbol=args.symbol)
    trainer = UnifiedTrainer(market_type=args.market, symbol=args.symbol)
    results = await trainer.run(save_models=True)
    logger.info("Training complete", results=results)

if __name__ == "__main__":
    asyncio.run(main())
