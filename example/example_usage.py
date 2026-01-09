"""
Example migration from inventa_model.py to core modules.

This file shows how to use the new organized structure.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import holidays
import math
from dotenv import load_dotenv
import os
from openai import OpenAI

# Import from core modules
from core import (
    # Schema
    ForecastResult,
    SalesPattern,
    StockAnalysis,
    
    # Features
    fill_missing_with_zero,
    remove_negative_sales,
    cap_outliers,
    add_lebaran_flags,
    build_calendar_features,
    build_future_calendar_features,
    analyze_sales_patterns,
    
    # Forecast
    EXOG_COLS,
    train_sarimax_all_products,
    hybrid_forecast,
    estimate_days_until_stockout,
    calculate_risk_and_urgency,
    decide_restock_action,
    calculate_business_priority,
    generate_portfolio_insights,
)

# Load environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load data
with open("/Users/rasya2121/Documents/code/projects/dinacom/test_doc.json") as f:
    data = json.load(f)

# Prepare initial dataframe
rows = []
for p in data['products']:
    for d in p['daily_sales']:
        rows.append({
            "product_id": p["product_id"],
            "product_name": p["product_name"],
            "unit": p["unit"],
            "date": d["date"],
            "qty_sold": d["qty"],
            "current_stock": p["stock"]["current_stock_on_hand"]
        })

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])

# Aggregate by product
df_product = (
    df.sort_values("date")
    .groupby(["product_id", "product_name", "unit", "current_stock"])
    .agg(
        daily_sales=("qty_sold", list),
        total_sales=("qty_sold", "sum"),
        avg_daily_sales=("qty_sold", "mean"),
        window_start=("date", "min"),
        window_end=("date", "max")
    )
    .reset_index()
)

# Clean data using core.features
df_product['daily_sales'] = df_product['daily_sales'].apply(fill_missing_with_zero)
df_product["daily_sales"] = df_product["daily_sales"].apply(remove_negative_sales)
df_product["daily_sales"] = df_product["daily_sales"].apply(cap_outliers)

# Create time series dataframe
rows = []
for _, r in df_product.iterrows():
    dates = pd.date_range(
        start=r["window_start"],
        periods=len(r["daily_sales"]),
        freq="D"
    )
    for d, qty in zip(dates, r["daily_sales"]):
        rows.append({
            "product_id": r["product_id"],
            "product_name": r["product_name"],
            "unit": r["unit"],
            "date": d,
            "qty": int(qty),
            "current_stock": r["current_stock"],
            "window_start": r["window_start"],
            "window_end": r["window_end"],
        })

df_ts = pd.DataFrame(rows)

# Add calendar features
df_ts["dow"] = df_ts["date"].dt.dayofweek
df_ts["is_weekend"] = df_ts["dow"].isin([5, 6]).astype(int)
df_ts["dow_sin"] = np.sin(2*np.pi*df_ts["dow"]/7)
df_ts["dow_cos"] = np.cos(2*np.pi*df_ts["dow"]/7)

# Add holiday features
id_holidays = holidays.Indonesia()
df_ts["is_holiday"] = df_ts["date"].dt.date.isin(id_holidays).astype(int)
df_ts["holiday_name"] = df_ts["date"].dt.date.map(lambda d: id_holidays.get(d)).fillna("")

# Add Lebaran flags using core.features
df_ts = add_lebaran_flags(df_ts, data["lebaran_date"])

# Train models using core.forecast
print("🚀 Training SARIMAX models...")
scores_df, results = train_sarimax_all_products(df_ts, verbose=True)

print("\n📊 Generating forecasts for all products...")
output_products = []

for pid in df_ts['product_id'].unique():
    g = df_ts[df_ts["product_id"] == pid].copy()
    current_stock = g["current_stock"].iloc[0] if len(g) > 0 else 0
    product_name = g["product_name"].iloc[0] if len(g) > 0 else pid
    unit = g["unit"].iloc[0] if len(g) > 0 else "unit"
    
    # Generate forecast using core.forecast
    forecast_result = hybrid_forecast(
        df_ts=df_ts,
        results=results,
        scores_df=scores_df,
        product_id=pid,
        horizon=14,
        use_ewma_fallback=True,
        holiday_dates=id_holidays.keys(),
        lebaran_date=data["lebaran_date"]
    )
    
    forecast_14 = [math.ceil(x) for x in forecast_result['forecast'][:14]]
    total_14 = sum(forecast_14)
    avg_14 = round(np.mean(forecast_14), 2)
    
    # Get WAPE
    wape_val = np.nan
    if isinstance(scores_df, pd.DataFrame) and "product_id" in scores_df.columns:
        tmp = scores_df[scores_df["product_id"].astype(str).str.strip() == str(pid).strip()]
        if not tmp.empty:
            wape_val = tmp.iloc[0].get("wape", np.nan)
    
    # Analyze using core functions
    days_until_stockout = estimate_days_until_stockout(current_stock, forecast_14)
    risk_info = calculate_risk_and_urgency(
        days_until_stockout=days_until_stockout,
        forecast_mean=avg_14,
        wape=wape_val,
        current_stock=current_stock
    )
    
    restock_decision = decide_restock_action(
        current_stock=current_stock,
        forecast_total_14=total_14,
        forecast_mean=avg_14,
        risk_level=risk_info["risk_level"],
        days_until_stockout=days_until_stockout,
        safety_stock_multiplier=1.5
    )
    
    # Pattern analysis using core.features
    sales_patterns = analyze_sales_patterns(df_ts, pid)
    
    product_obj = {
        "product_id": str(pid),
        "product_name": product_name,
        "unit": unit,
        "current_stock": int(current_stock),
        
        "forecast": {
            "horizon_days": 14,
            "daily": forecast_14,
            "total_demand": total_14,
            "average_per_day": avg_14,
            "method": forecast_result["method"],
            "confidence": forecast_result["confidence"]
        },
        
        "stock_analysis": {
            "days_until_stockout": days_until_stockout,
            "risk_level": risk_info["risk_level"],
            "urgency_score": risk_info["urgency_score"],
            "forecast_reliability": "HIGH" if pd.notna(wape_val) and wape_val < 0.15 else "MEDIUM" if pd.notna(wape_val) and wape_val < 0.30 else "LOW"
        },
        
        "recommendation": {
            "action": restock_decision["action"],
            "quantity_range": {
                "min": restock_decision["qty_min"],
                "max": restock_decision["qty_max"]
            },
            "reason": restock_decision["reason"]
        },
        
        "business_insights": {
            "sales_patterns": sales_patterns
        }
    }
    
    # Calculate priority using core.forecast
    priority_score = calculate_business_priority(product_obj, sales_patterns)
    
    if priority_score >= 80:
        priority_tier = "CRITICAL"
    elif priority_score >= 60:
        priority_tier = "HIGH"
    elif priority_score >= 40:
        priority_tier = "MEDIUM"
    else:
        priority_tier = "LOW"
    
    product_obj["business_priority"] = {
        "priority_score": priority_score,
        "priority_tier": priority_tier
    }
    
    output_products.append(product_obj)
    
    print(f"[{pid}] {product_name} - Priority: {priority_tier} ({priority_score}/100)")

# Generate portfolio insights using core.forecast
print("\n📈 Generating portfolio insights...")
portfolio_insights = generate_portfolio_insights(output_products)

# Create final output
final_json = {
    "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_version": "hybrid_sarimax_ewma_v3.0",
    "total_products": len(output_products),
    "portfolio_insights": portfolio_insights,
    "products": sorted(
        output_products, 
        key=lambda x: x['business_priority']['priority_score'], 
        reverse=True
    )
}

# Save output
output_file = "forecast_output_new.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_json, f, indent=2, ensure_ascii=False)

print(f"\n✅ Output saved to: {output_file}")
print(f"📦 Total products: {final_json['total_products']}")
print(f"🚨 High risk: {portfolio_insights['summary']['high_risk_count']} products")
print(f"📈 Growing: {portfolio_insights['trends']['growing_products']} products")
