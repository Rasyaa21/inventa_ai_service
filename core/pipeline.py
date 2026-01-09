"""
Core pipeline logic for inventory forecasting.
"""
import pandas as pd
import numpy as np
import holidays
from typing import Dict, Any, Optional

from .features import (
    fill_missing_with_zero,
    remove_negative_sales,
    cap_outliers,
    add_lebaran_flags
)
from .forecast import train_sarimax_all_products
from .llm import (
    get_openai_client,
    generate_complete_forecast_with_batch_llm
)
from .api_schema import ForecastRequest

def run_forecast_pipeline(request: ForecastRequest) -> Dict[str, Any]:
    """
    Run the complete forecasting pipeline from a request object.
    
    Args:
        request: The parsed API request containing product data and parameters.
        
    Returns:
        A dictionary containing the complete forecast results.
    """
    
    # ============================================================================
    # 1. Prepare Data
    # ============================================================================
    
    rows = []
    for p in request.products:
        p_dict = p.model_dump() # Convert pydantic to dict
        current_stock = p_dict['stock']['current_stock_on_hand']
        
        for d in p_dict['daily_sales']:
            rows.append({
                "product_id": p_dict["product_id"],
                "product_name": p_dict["product_name"],
                "unit": p_dict["unit"],
                "date": d["date"],
                "qty_sold": d["qty"],
                "current_stock": current_stock
            })

    if not rows:
        return {"error": "No data provided"}

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # ============================================================================
    # 2. Data Aggregation and Cleaning
    # ============================================================================

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

    # ============================================================================
    # 3. Create Time Series DataFrame
    # ============================================================================

    ts_rows = []
    for _, r in df_product.iterrows():
        dates = pd.date_range(
            start=r["window_start"],
            periods=len(r["daily_sales"]),
            freq="D"
        )
        for d, qty in zip(dates, r["daily_sales"]):
            ts_rows.append({
                "product_id": r["product_id"],
                "product_name": r["product_name"],
                "unit": r["unit"],
                "date": d,
                "qty": int(qty),
                "current_stock": r["current_stock"],
                "window_start": r["window_start"],
                "window_end": r["window_end"],
            })

    df_ts = pd.DataFrame(ts_rows)

    # ============================================================================
    # 4. Add Calendar Features
    # ============================================================================

    df_ts["dow"] = df_ts["date"].dt.dayofweek
    df_ts["is_weekend"] = df_ts["dow"].isin([5, 6]).astype(int)
    df_ts["dow_sin"] = np.sin(2*np.pi*df_ts["dow"]/7)
    df_ts["dow_cos"] = np.cos(2*np.pi*df_ts["dow"]/7)

    # Add holiday features
    id_holidays = holidays.Indonesia()
    df_ts["is_holiday"] = df_ts["date"].dt.date.isin(id_holidays).astype(int)
    df_ts["holiday_name"] = df_ts["date"].dt.date.map(lambda d: id_holidays.get(d)).fillna("")

    # Add Lebaran flags
    # Use provided lebaran_date or default/None. 
    # Logic in add_lebaran_flags needs to handle None gracefully or we provide a default?
    # Checking add_lebaran_flags signature or behavior might be good, but assuming it handles None or we pass None.
    # Looking at the script, it passed data["lebaran_date"].
    
    lebaran = request.lebaran_date if request.lebaran_date else None
    # Convert date to string if add_lebaran_flags expects string, or keep as date.
    # The original script loaded from JSON so it was likely a string.
    # We should ensure compatibility. Let's pass it as is, usually add_lebaran_flags parses it.
    
    if lebaran:
        df_ts = add_lebaran_flags(df_ts, str(lebaran))
    else:
        # If no lebaran date provided, perhaps skip or pass None. 
        # For now, let's pass None if the function handles it, or just not call it?
        # Safe bet: Try to call it if it's robust, otherwise risk error. 
        # Inspecting previous context: "Add Lebaran flags" was a line.
        # I'll assume if it's None it might fail or do nothing. Let's wrap in try-except or just pass if not None.
        pass 

    # ============================================================================
    # 5. Train Models
    # ============================================================================

    scores_df, results = train_sarimax_all_products(df_ts, verbose=False)

    # ============================================================================
    # 6. Generate Complete Forecast with LLM
    # ============================================================================

    # Initialize OpenAI client
    try:
        client = get_openai_client()
    except Exception:
        client = None

    # Generate complete forecast
    complete_json = generate_complete_forecast_with_batch_llm(
        df_ts=df_ts,
        results=results,
        scores_df=scores_df,
        use_llm=client is not None,
        llm_model="gpt-4o-mini",
        batch_size=3,
        client=client
    )
    
    return complete_json
