"""
Dinacom AI - Inventory Forecasting Core Module
"""

# Schema exports
from .schema import (
    ForecastResult,
    SalesPattern,
    StockAnalysis,
    RestockRecommendation,
    BusinessPriority,
    ProductForecast,
)

# Features exports
from .features import (
    fill_missing_with_zero,
    remove_negative_sales,
    cap_outliers,
    add_lebaran_flags,
    build_calendar_features,
    build_future_calendar_features,
    compute_calendar_multipliers_from_g,
    apply_calendar_adjustment,
    apply_calendar_adjustment_to_test,
    prepare_sarimax_data,
    prepare_sarimax_product_df,
    analyze_sales_patterns,
)

# Forecast exports
from .forecast import (
    EXOG_COLS,
    mae,
    rmse,
    wape,
    select_ewma_alpha,
    ewma_forecast,
    ewma_pandas,
    ewma_with_trend,
    train_sarimax_all_products,
    should_use_ewma_fallback,
    blend_forecasts,
    hybrid_forecast,
    estimate_days_until_stockout,
    calculate_risk_and_urgency,
    decide_restock_action,
    calculate_business_priority,
    generate_portfolio_insights,
)

# LLM exports
from .llm import (
    get_llm_client,
    generate_batch_llm_analysis,
    generate_rule_based_fallback,
    generate_complete_forecast_with_batch_llm,
)

__all__ = [
    # Schema
    "ForecastResult",
    "SalesPattern",
    "StockAnalysis",
    "RestockRecommendation",
    "BusinessPriority",
    "ProductForecast",
    
    # Features
    "fill_missing_with_zero",
    "remove_negative_sales",
    "cap_outliers",
    "add_lebaran_flags",
    "build_calendar_features",
    "build_future_calendar_features",
    "compute_calendar_multipliers_from_g",
    "apply_calendar_adjustment",
    "apply_calendar_adjustment_to_test",
    "prepare_sarimax_data",
    "prepare_sarimax_product_df",
    "analyze_sales_patterns",
    
    # Forecast
    "EXOG_COLS",
    "mae",
    "rmse",
    "wape",
    "select_ewma_alpha",
    "ewma_forecast",
    "ewma_pandas",
    "ewma_with_trend",
    "train_sarimax_all_products",
    "should_use_ewma_fallback",
    "blend_forecasts",
    "hybrid_forecast",
    "estimate_days_until_stockout",
    "calculate_risk_and_urgency",
    "decide_restock_action",
    "calculate_business_priority",
    "generate_portfolio_insights",
    
    # LLM
    "get_llm_client",
    "generate_batch_llm_analysis",
    "generate_rule_based_fallback",
    "generate_complete_forecast_with_batch_llm",
]
