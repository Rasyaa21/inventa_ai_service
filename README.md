# Dinacom AI - Inventory Forecasting System

## Project Structure

```
dinacom-ai/
├── core/                      # Core forecasting modules
│   ├── __init__.py           # Module exports
│   ├── schema.py             # Data models and schemas
│   ├── features.py           # Feature engineering functions
│   ├── forecast.py           # Forecasting algorithms
│   └── llm.py                # LLM integration (OpenAI)
├── inventa_model.py          # Original notebook conversion (legacy)
├── inventa_model.ipynb       # Jupyter notebook (development)
├── app.py                    # Main application
├── run_pipeline.py           # Complete pipeline with LLM
├── example_usage.py          # Example usage without LLM
├── .env                      # Environment variables (API keys)
└── requirements.txt          # Python dependencies
```

## Module Organization

### `core/schema.py`
Data models and type definitions:
- `ForecastResult`: Forecast output with metadata
- `SalesPattern`: Sales pattern analysis
- `StockAnalysis`: Risk assessment and stockout prediction
- `RestockRecommendation`: Restock action recommendations
- `BusinessPriority`: Priority scoring
- `ProductForecast`: Complete product forecast structure

### `core/features.py`
Feature engineering and preprocessing:
- Data cleaning: `fill_missing_with_zero()`, `remove_negative_sales()`, `cap_outliers()`
- Calendar features: `build_calendar_features()`, `build_future_calendar_features()`
- Lebaran/holiday flags: `add_lebaran_flags()`
- Calendar adjustments: `compute_calendar_multipliers_from_g()`, `apply_calendar_adjustment()`
- Sales pattern analysis: `analyze_sales_patterns()`

### `core/forecast.py`
Forecasting models and business logic:
- **Metrics**: `mae()`, `rmse()`, `wape()`
- **EWMA Models**: `ewma_forecast()`, `ewma_pandas()`, `ewma_with_trend()`
- **SARIMAX**: `train_sarimax_all_products()`
- **Hybrid Forecasting**: `hybrid_forecast()`, `blend_forecasts()`
- **Stock Analysis**: `estimate_days_until_stockout()`, `calculate_risk_and_urgency()`
- **Recommendations**: `decide_restock_action()`, `calculate_business_priority()`
- **Portfolio Insights**: `generate_portfolio_insights()`

### `core/llm.py`
LLM integration for AI-powered recommendations:
- **OpenAI Client**: `get_openai_client()`
- **Batch Analysis**: `generate_batch_llm_analysis()` - Efficient batch processing
- **Fallback**: `generate_rule_based_fallback()` - Rule-based reasoning when LLM fails
- **Complete Pipeline**: `generate_complete_forecast_with_batch_llm()` - End-to-end with LLM

## Usage Example

```python
from core import (
    train_sarimax_all_products,
    hybrid_forecast,
    analyze_sales_patterns,
    calculate_business_priority,
    generate_batch_llm_analysis
)

# Train models
scores_df, results = train_sarimax_all_products(df_ts)

# Generate hybrid forecast
forecast_result = hybrid_forecast(
    df_ts=df_ts,
    results=results,
    scores_df=scores_df,
    product_id="P001",
    horizon=14
)

# Analyze patterns
patterns = analyze_sales_patterns(df_ts, product_id="P001")

# Calculate priority
priority = calculate_business_priority(product_data, patterns)

# Generate AI insights (batch processing)
llm_results, tokens = generate_batch_llm_analysis(
    products_data=output_products,
    model="gpt-4o-mini",
    batch_size=3
)
```

## LLM Features

### Batch Processing
Process multiple products in a single API call for efficiency:
- Configurable batch size (default: 3 products per batch)
- Automatic rate limiting with 1-second delays
- Smart prioritization (HIGH risk products first)

### Intelligent Fallback
- Automatic fallback to rule-based reasoning if LLM fails
- No disruption to pipeline execution
- Clear indication in output (`model_used` field)

### Output Format
Each product gets AI-powered insights:

```json
{
  "ai_insights": {
    "reasoning": "**Kesimpulan:** URGENT - Stok akan habis dalam 3 hari!\n**Tindakan:** Segera restock 100-150 unit...\n**Impact:** Kehilangan penjualan jika tidak bertindak.",
    "model": "gpt-4o-mini",
    "generated_at": "2026-01-09 15:30:00"
  }
}
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

## Quick Start

### Basic Usage (without LLM)

```python
from core import train_sarimax_all_products, hybrid_forecast

# Train models
scores_df, results = train_sarimax_all_products(df_ts)

# Generate forecast
forecast = hybrid_forecast(
    df_ts=df_ts,
    results=results,
    scores_df=scores_df,
    product_id="P001",
    horizon=14
)
```

### Complete Pipeline with LLM

```bash
# Run the complete pipeline
python run_pipeline.py
```

Or use in your code:

```python
from dotenv import load_dotenv
from core import (
    train_sarimax_all_products,
    get_openai_client,
    generate_complete_forecast_with_batch_llm
)

load_dotenv()

# Train models
scores_df, results = train_sarimax_all_products(df_ts)

# Initialize OpenAI client
client = get_openai_client()

# Generate complete forecast with AI insights
complete_json = generate_complete_forecast_with_batch_llm(
    df_ts=df_ts,
    results=results,
    scores_df=scores_df,
    use_llm=True,
    llm_model="gpt-4o-mini",
    batch_size=3,
    client=client
)
```

## Configuration

Default exogenous features (configurable via `EXOG_COLS`):
- `is_weekend`: Weekend indicator
- `dow_sin`, `dow_cos`: Day of week cyclical encoding
- `is_holiday`: Holiday indicator
- `is_pre_lebaran_30d`: 30 days before Lebaran
- `is_lebaran`: Lebaran day
- `is_post_lebaran_14d`: 14 days after Lebaran

## Model Details

### Hybrid Forecasting Strategy
1. **SARIMAX**: Seasonal ARIMA with exogenous variables
2. **EWMA with Trend**: Exponentially weighted moving average with trend adjustment
3. **Decision Logic**: Automatically selects or blends models based on:
   - Historical accuracy (WAPE)
   - Data availability
   - Trend strength
   - Forecast volatility

### Priority Scoring (0-100)
- **40 points**: Urgency (based on stockout risk)
- **20 points**: Demand volume
- **15 points**: Volatility impact
- **15 points**: Growth trend
- **10 points**: Forecast reliability

Priority Tiers:
- **CRITICAL**: Score ≥ 80
- **HIGH**: Score ≥ 60
- **MEDIUM**: Score ≥ 40
- **LOW**: Score < 40
