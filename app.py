import logging
import uvicorn
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("settlebot_app")

if __name__ == "__main__":
    logger.info(
        f"Starting SettleBot API on {settings.api.host}:{settings.api.port}"
    )
    logger.info(
        "Settlement Assistant for International Students in Nairobi, Kenya"
    )

    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level="info",
    )
