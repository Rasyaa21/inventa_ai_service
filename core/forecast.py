"""
Forecasting functions including SARIMAX, EWMA, and hybrid approaches.
"""
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict, Any, Optional, Tuple, List

from .features import (
    build_calendar_features,
    build_future_calendar_features,
    prepare_sarimax_data,
    prepare_sarimax_product_df
)


# Exogenous columns configuration
EXOG_COLS = [
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "is_pre_lebaran_30d",
    "is_lebaran",
    "is_post_lebaran_14d",
]


# ============================================================================
# Metrics
# ============================================================================

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred, eps=1e-9):
    """Weighted Absolute Percentage Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))


# ============================================================================
# EWMA Functions
# ============================================================================

def select_ewma_alpha(n_obs, recent_trend=None):
    """Select EWMA alpha based on observation count and trend."""
    if n_obs <= 21:
        base_alpha = 0.3
    elif n_obs <= 40:
        base_alpha = 0.25
    else:
        base_alpha = 0.2
    
    if recent_trend is not None:
        if abs(recent_trend) > 0.2:
            base_alpha = min(base_alpha + 0.1, 0.4)
    
    return base_alpha


def ewma_forecast(series, alpha=0.3, horizon=14):
    """Basic EWMA forecast."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    
    ewma_vals = np.zeros(n)
    ewma_vals[0] = series[0]
    
    for t in range(1, n):
        ewma_vals[t] = alpha * series[t] + (1 - alpha) * ewma_vals[t-1]
    
    last_ewma = ewma_vals[-1]
    forecast = np.full(horizon, last_ewma)
    
    return {
        'fitted': ewma_vals,
        'forecast': forecast,
        'last_value': last_ewma
    }


def ewma_pandas(series, alpha=0.3, horizon=14):
    """EWMA using pandas."""
    s = pd.Series(series, dtype=float)
    ewma_vals = s.ewm(alpha=alpha, adjust=False).mean()
    last_ewma = ewma_vals.iloc[-1]
    forecast = np.full(horizon, last_ewma)
    
    return {
        'fitted': ewma_vals.to_numpy(),
        'forecast': forecast,
        'last_value': last_ewma
    }


def ewma_with_trend(series, alpha=0.3, horizon=14):
    """EWMA with trend adjustment for better forecasting."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    
    if n < 3:
        return ewma_pandas(series, alpha=alpha, horizon=horizon)
    
    s = pd.Series(series)
    ewma_vals = s.ewm(alpha=alpha, adjust=False).mean()
    last_ewma = ewma_vals.iloc[-1]
    
    lookback = min(7, n)
    recent_vals = series[-lookback:]
    
    # Simple linear trend
    x = np.arange(lookback)
    slope, intercept = np.polyfit(x, recent_vals, 1)
    
    # Project with trend
    forecast = []
    for h in range(1, horizon + 1):
        # Dampen trend over time
        damping = 0.9 ** h
        projected = last_ewma + (slope * h * damping)
        
        # Safety bounds
        projected = max(projected, 0)
        projected = min(projected, last_ewma * 3)
        
        forecast.append(projected)
    
    return {
        'fitted': ewma_vals.to_numpy(),
        'forecast': np.array(forecast),
        'last_value': last_ewma,
        'trend_slope': slope
    }


# ============================================================================
# SARIMAX Training
# ============================================================================

def train_sarimax_all_products(
    df_ts,
    order=(1,1,1),
    seasonal_order=(1,0,1,7),
    product_col="product_id",
    date_col="date",
    min_obs=30,
    verbose=True,
    exog_cols=None
):
    """Train SARIMAX models for all products."""
    if exog_cols is None:
        exog_cols = EXOG_COLS
        
    df = df_ts.copy()
    df[product_col] = df[product_col].astype(str).str.strip()
    df[date_col] = pd.to_datetime(df[date_col])

    results = {}
    rows = []

    product_ids = df[product_col].dropna().unique().tolist()

    for pid in product_ids:
        g = df[df[product_col] == pid].sort_values(date_col)

        last_date = g[date_col].max()
        n_obs = len(g)

        pname = g["product_name"].iloc[0] if "product_name" in g.columns and n_obs > 0 else pid
        unit = g["unit"].iloc[0] if "unit" in g.columns and n_obs > 0 else None

        if n_obs < min_obs:
            msg = f"skip: n_obs={n_obs} < min_obs={min_obs}"
            if verbose:
                print(f"[{pid}] {msg}")
            results[pid] = {
                "res": None,
                "scaler": None,
                "exog_cols": None,
                "last_date": last_date,
                "n_obs": n_obs,
                "error": msg
            }
            rows.append({
                "product_id": pid,
                "product_name": pname,
                "unit": unit,
                "n_obs": n_obs,
                "last_date": last_date,
                "aic": np.nan,
                "bic": np.nan,
                "hqic": np.nan,
                "llf": np.nan,
                "status": "SKIPPED",
                "error": msg
            })
            continue

        try:
            y, X, scaler = prepare_sarimax_data(df_ts, pid, exog_cols)

            if len(y) != len(X):
                raise ValueError(f"len(y)={len(y)} != len(X)={len(X)}")

            X = X.apply(pd.to_numeric, errors="coerce")
            if X.isna().any().any():
                X = X.fillna(0.0)

            model = SARIMAX(
                y,
                exog=X,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)

            results[pid] = {
                "res": res,
                "scaler": scaler,
                "exog_cols": list(X.columns),
                "last_date": last_date,
                "n_obs": len(y),
                "error": None
            }

            rows.append({
                "product_id": pid,
                "product_name": pname,
                "unit": unit,
                "n_obs": len(y),
                "last_date": last_date,
                "aic": float(res.aic),
                "bic": float(res.bic),
                "hqic": float(res.hqic),
                "llf": float(res.llf),
                "status": "OK",
                "error": ""
            })

            if verbose:
                print(f"[{pid}] OK | n={len(y)} | AIC={res.aic:.2f} | BIC={res.bic:.2f}")

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"[{pid}] FAIL | {msg}")

            results[pid] = {
                "res": None,
                "scaler": None,
                "exog_cols": None,
                "last_date": last_date,
                "n_obs": n_obs,
                "error": msg
            }
            rows.append({
                "product_id": pid,
                "product_name": pname,
                "unit": unit,
                "n_obs": n_obs,
                "last_date": last_date,
                "aic": np.nan,
                "bic": np.nan,
                "hqic": np.nan,
                "llf": np.nan,
                "status": "FAIL",
                "error": msg
            })

    scores_df = pd.DataFrame(rows).sort_values(["status", "aic"], ascending=[True, True])
    return scores_df, results


# ============================================================================
# Hybrid Forecast
# ============================================================================

def should_use_ewma_fallback(sarimax_wape, sarimax_forecast, ewma_forecast, threshold=0.5):
    """Decide whether to use EWMA fallback."""
    if pd.notna(sarimax_wape) and sarimax_wape > threshold:
        return 'ewma'
    
    sarimax_mean = np.mean(sarimax_forecast)
    ewma_mean = np.mean(ewma_forecast)
    
    if sarimax_mean > 3 * ewma_mean and ewma_mean > 1:
        return 'blend'
    
    if np.any(sarimax_forecast < 0):
        return 'ewma'
    
    return 'sarimax'


def blend_forecasts(sarimax_fc, ewma_fc, weight=0.7):
    """Blend SARIMAX and EWMA forecasts."""
    sarimax_fc = np.asarray(sarimax_fc, dtype=float)
    ewma_fc = np.asarray(ewma_fc, dtype=float)
    return weight * sarimax_fc + (1 - weight) * ewma_fc


def hybrid_forecast(
    df_ts,
    results,
    scores_df,
    product_id,
    horizon=14,
    use_ewma_fallback=True,
    blend_threshold=0.5,
    holiday_dates=None,
    lebaran_date=None
):
    """Enhanced hybrid forecast with trend-aware EWMA."""
    pid = str(product_id).strip()
    
    g = df_ts[df_ts["product_id"] == pid].copy()
    g = g.sort_values("date")
    y_hist = g["qty"].astype(float).to_numpy()
    
    n_obs = len(y_hist)
    
    # Handle very short data
    if n_obs < 7:
        avg_sales = np.mean(y_hist)
        simple_fc = np.full(horizon, avg_sales)
        return {
            'forecast': simple_fc,
            'method': 'simple_average',
            'confidence': 'low',
            'reason': f'Only {n_obs} days of data - using simple average'
        }
    
    # Trend-aware EWMA
    alpha = select_ewma_alpha(n_obs)
    ewma_result = ewma_with_trend(y_hist, alpha=alpha, horizon=horizon)
    ewma_fc = ewma_result['forecast']
    trend_slope = ewma_result.get('trend_slope', 0)
    
    # SARIMAX forecast
    r = results.get(pid)
    if r is None or r["res"] is None:
        return {
            'forecast': ewma_fc,
            'method': 'ewma_with_trend',
            'confidence': 'medium' if n_obs >= 14 else 'low',
            'reason': 'SARIMAX not available'
        }
    
    last_date = pd.to_datetime(r["last_date"])
    exog_cols = list(r["exog_cols"])
    
    future_exog = build_future_calendar_features(
        start_date=last_date + pd.Timedelta(days=1),
        horizon=horizon,
        holiday_dates=holiday_dates,
        lebaran_date=lebaran_date
    )[exog_cols]
    
    if r.get("scaler") is not None:
        future_exog_scaled = pd.DataFrame(
            r["scaler"].transform(future_exog),
            columns=exog_cols
        )
    else:
        future_exog_scaled = future_exog
    
    sarimax_fc = r["res"].forecast(steps=horizon, exog=future_exog_scaled)
    sarimax_fc = np.asarray(sarimax_fc, dtype=float)
    
    # Get backtest WAPE
    wape_val = np.nan
    if isinstance(scores_df, pd.DataFrame):
        tmp = scores_df[scores_df["product_id"].astype(str).str.strip() == pid]
        if not tmp.empty:
            wape_val = tmp.iloc[0].get("wape", np.nan)
    
    if not use_ewma_fallback:
        return {
            'forecast': sarimax_fc,
            'method': 'sarimax_only',
            'confidence': 'medium'
        }
    
    # Enhanced decision logic
    decision = should_use_ewma_fallback(wape_val, sarimax_fc, ewma_fc, threshold=blend_threshold)
    
    # Additional check for strong trends
    sarimax_std = np.std(sarimax_fc)
    if abs(trend_slope) > 0.5 and sarimax_std < 1.0:
        decision = 'ewma'
    
    if decision == 'ewma':
        return {
            'forecast': ewma_fc,
            'method': 'ewma_override',
            'confidence': 'medium',
            'reason': f'SARIMAX WAPE={wape_val*100:.1f}% or trend detected'
        }
    elif decision == 'blend':
        weight = 0.4 if abs(trend_slope) > 0.3 else 0.6
        blended = blend_forecasts(sarimax_fc, ewma_fc, weight=weight)
        return {
            'forecast': blended,
            'method': 'hybrid_blend',
            'confidence': 'high',
            'reason': f'Blended: {int(weight*100)}% SARIMAX, {int((1-weight)*100)}% EWMA'
        }
    else:
        return {
            'forecast': sarimax_fc,
            'method': 'sarimax_trusted',
            'confidence': 'high'
        }


# ============================================================================
# Stock Analysis and Recommendations
# ============================================================================

def estimate_days_until_stockout(current_stock, daily_forecast):
    """Calculate days until stockout."""
    cumulative = 0
    for day, qty in enumerate(daily_forecast, start=1):
        cumulative += qty
        if cumulative >= current_stock:
            return day
    return None


def calculate_risk_and_urgency(days_until_stockout, forecast_mean, wape, current_stock):
    """Calculate risk level and urgency score."""
    if days_until_stockout is None:
        risk_level = "LOW"
        urgency_score = 10
    elif days_until_stockout <= 3:
        risk_level = "HIGH"
        urgency_score = 70
    elif days_until_stockout <= 7:
        risk_level = "MEDIUM"
        urgency_score = 50
    else:
        risk_level = "LOW"
        urgency_score = 30
    
    return {
        "risk_level": risk_level,
        "urgency_score": urgency_score
    }


def decide_restock_action(
    current_stock, 
    forecast_total_14, 
    forecast_mean, 
    risk_level, 
    days_until_stockout, 
    safety_stock_multiplier=1.5
):
    """Decide restock action with quantity range."""
    safety_stock = int(forecast_mean * 7 * safety_stock_multiplier)
    
    if risk_level == "HIGH":
        qty_min = int((forecast_total_14 + safety_stock) - current_stock)
        qty_max = int(qty_min * 1.5)
        return {
            "action": "RESTOCK",
            "qty_min": max(qty_min, 0),
            "qty_max": qty_max,
            "reason": f"HIGH RISK: Stock runs out in {days_until_stockout} days. Predicted demand exceeds current stock."
        }
    
    elif risk_level == "MEDIUM":
        qty_min = int((forecast_total_14 * 0.7 + safety_stock) - current_stock)
        qty_max = int(qty_min * 1.5)
        return {
            "action": "RESTOCK",
            "qty_min": max(qty_min, 0),
            "qty_max": qty_max,
            "reason": f"MEDIUM RISK: Stock runs out in {days_until_stockout} days. Moderate restock needed."
        }
    
    else:
        if current_stock < safety_stock:
            qty_min = int(safety_stock - current_stock)
            qty_max = int(qty_min * 1.2)
            return {
                "action": "RESTOCK",
                "qty_min": qty_min,
                "qty_max": qty_max,
                "reason": "LOW RISK: Stock sufficient for now, but restock recommended."
            }
        else:
            return {
                "action": "WAIT",
                "qty_min": 0,
                "qty_max": 0,
                "reason": "Stock sufficient for forecast period. Monitor and wait."
            }


def calculate_business_priority(product_data, sales_patterns):
    """Calculate business priority score (0-100)."""
    score = 0
    
    # 1. Urgency Score (40 points)
    urgency = product_data['stock_analysis']['urgency_score']
    score += (urgency / 100) * 40
    
    # 2. Demand Volume (20 points)
    avg_demand = product_data['forecast']['average_per_day']
    if avg_demand > 100:
        score += 20
    elif avg_demand > 50:
        score += 15
    elif avg_demand > 20:
        score += 10
    elif avg_demand > 5:
        score += 5
    
    # 3. Volatility Impact (15 points)
    if sales_patterns['volatility'] == "high":
        score += 15
    elif sales_patterns['volatility'] == "medium":
        score += 10
    else:
        score += 5
    
    # 4. Growth Trend (15 points)
    if sales_patterns['trend'] == "growing":
        score += 15
    elif sales_patterns['trend'] == "stable":
        score += 10
    elif sales_patterns['trend'] == "declining":
        score += 5
    
    # 5. Forecast Reliability (10 points)
    reliability = product_data['stock_analysis']['forecast_reliability']
    if reliability == "HIGH":
        score += 10
    elif reliability == "MEDIUM":
        score += 6
    else:
        score += 3
    
    return int(min(score, 100))


def generate_portfolio_insights(all_products):
    """Generate high-level business insights from portfolio."""
    high_risk = [p for p in all_products if p['stock_analysis']['risk_level'] == 'HIGH']
    medium_risk = [p for p in all_products if p['stock_analysis']['risk_level'] == 'MEDIUM']
    growing = [p for p in all_products if p['business_insights']['sales_patterns']['trend'] == 'growing']
    declining = [p for p in all_products if p['business_insights']['sales_patterns']['trend'] == 'declining']
    high_volatility = [p for p in all_products if p['business_insights']['sales_patterns']['volatility'] == 'high']
    
    total_restock_value = sum(
        p['recommendation']['quantity_range']['max'] 
        for p in all_products 
        if p['recommendation']['action'] == 'RESTOCK'
    )
    
    insights = {
        "summary": {
            "total_products": len(all_products),
            "high_risk_count": len(high_risk),
            "medium_risk_count": len(medium_risk),
            "restock_needed_count": sum(1 for p in all_products if p['recommendation']['action'] == 'RESTOCK'),
            "estimated_total_restock_qty": total_restock_value
        },
        
        "trends": {
            "growing_products": len(growing),
            "declining_products": len(declining),
            "high_volatility_products": len(high_volatility)
        },
        
        "priority_actions": [
            {
                "product_name": p['product_name'],
                "risk_level": p['stock_analysis']['risk_level'],
                "days_left": p['stock_analysis']['days_until_stockout'],
                "priority_score": p['business_priority']['priority_score'],
                "recommended_qty": f"{p['recommendation']['quantity_range']['min']}-{p['recommendation']['quantity_range']['max']} {p['unit']}"
            }
            for p in sorted(all_products, key=lambda x: x['business_priority']['priority_score'], reverse=True)[:5]
        ],
        
        "risk_distribution": {
            "HIGH": len(high_risk),
            "MEDIUM": len(medium_risk),
            "LOW": len(all_products) - len(high_risk) - len(medium_risk)
        }
    }
    
    return insights
