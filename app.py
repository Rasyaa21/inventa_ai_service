from fastapi import FastAPI, HTTPException
from core.api_schema import ForecastRequest
from core.pipeline import run_forecast_pipeline
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dinacom AI Forecasting API",
    description="API for inventory forecasting and replenishment recommendations.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Dinacom AI Forecasting"}

@app.post("/forecast")
def generate_forecast(request: ForecastRequest):
    """
    Generate forecasts for a list of products.
    """
    try:
        logger.info(f"Received forecast request for {len(request.products)} products")
        result = run_forecast_pipeline(request)
        return result
    except Exception as e:
        logger.error(f"Error processing forecast: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
