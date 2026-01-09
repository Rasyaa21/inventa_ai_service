"""
Data schemas and models for inventory forecasting system.
"""
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass


@dataclass
class ForecastResult:
    """Forecast result with metadata."""
    forecast: List[float]
    method: str
    confidence: str
    reason: Optional[str] = None
    fitted: Optional[List[float]] = None
    last_value: Optional[float] = None
    trend_slope: Optional[float] = None


@dataclass
class SalesPattern:
    """Sales pattern analysis."""
    trend: Literal["growing", "declining", "stable", "insufficient_data"]
    volatility: Literal["high", "medium", "low", "unknown"]
    weekend_effect: Optional[Literal["strong_increase", "strong_decrease", "neutral"]]
    growth_rate: Optional[float]
    seasonality: Literal["detected", "weak", "unknown"]
    avg_daily_sales: float
    coefficient_of_variation: float


@dataclass
class StockAnalysis:
    """Stock analysis with risk assessment."""
    days_until_stockout: Optional[int]
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]
    urgency_score: int
    forecast_reliability: Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class RestockRecommendation:
    """Restock recommendation with quantity range."""
    action: Literal["RESTOCK", "WAIT"]
    qty_min: int
    qty_max: int
    reason: str


@dataclass
class BusinessPriority:
    """Business priority scoring."""
    priority_score: int
    priority_tier: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]


@dataclass
class ProductForecast:
    """Complete product forecast with all analyses."""
    product_id: str
    product_name: str
    unit: str
    current_stock: int
    forecast: Dict[str, Any]
    stock_analysis: Dict[str, Any]
    recommendation: Dict[str, Any]
    business_insights: Dict[str, Any]
    business_priority: Dict[str, Any]
    ai_insights: Optional[Dict[str, Any]] = None
