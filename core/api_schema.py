"""
Pydantic Schemas for API requests and responses.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import date


class DailySales(BaseModel):
    date: date
    qty: int


class StockInfo(BaseModel):
    current_stock_on_hand: int


class ProductData(BaseModel):
    product_id: str
    product_name: str
    unit: Optional[str] = "pcs"
    stock: StockInfo
    daily_sales: List[DailySales]


class ForecastRequest(BaseModel):
    products: List[ProductData]
    lebaran_date: Optional[date] = None
