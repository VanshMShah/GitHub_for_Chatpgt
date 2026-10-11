import asyncio
import logging
from src.ats.agent import BackgroundAgent
from src.ats.engine import ATSEngine
from src.ats.store import Store
from src.ats.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

async def main():
    engine = ATSEngine(Store(settings.db_path))
    agent = BackgroundAgent(engine)
    try:
        await agent.run()
    finally:
        agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
