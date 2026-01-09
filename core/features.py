"""
Feature engineering functions for inventory forecasting.
"""
import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler


def fill_missing_with_zero(daily_sales):
    """Fill missing sales values with zero."""
    return [0 if x is None else x for x in daily_sales]


def remove_negative_sales(daily_sales):
    """Remove negative sales values by setting them to zero."""
    return [max(0, x) for x in daily_sales]


def cap_outliers(daily_sales, factor=4):
    """Cap outliers at factor times the mean."""
    arr = np.array(daily_sales)
    avg = arr.mean()
    cap = avg * factor
    return [min(x, cap) for x in arr]


def add_lebaran_flags(df, lebaran_date):
    """Add Lebaran-related flags to dataframe."""
    df = df.copy()
    d = pd.to_datetime(df["date"]).dt.normalize()
    lebaran_date = pd.to_datetime(lebaran_date).normalize()

    df["is_pre_lebaran_30d"] = (
        (d >= lebaran_date - pd.Timedelta(days=30)) &
        (d < lebaran_date)
    ).astype(int)

    df["is_lebaran"] = (d == lebaran_date).astype(int)

    df["is_post_lebaran_14d"] = (
        (d > lebaran_date) &
        (d <= lebaran_date + pd.Timedelta(days=14))
    ).astype(int)

    return df


def build_calendar_features(
    dates,
    holiday_dates=None,
    lebaran_date=None
):
    """Build calendar features for given dates."""
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    dow = d.dt.dayofweek

    is_weekend = (dow >= 5).astype(int)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    # Holiday features
    holiday_set = set(pd.to_datetime(list(holiday_dates)).normalize()) if holiday_dates is not None else set()
    is_holiday = d.isin(holiday_set).astype(int)

    # Lebaran features
    is_pre_lebaran_30d = np.zeros(len(d), dtype=int)
    is_lebaran = np.zeros(len(d), dtype=int)
    is_post_lebaran_14d = np.zeros(len(d), dtype=int)

    if lebaran_date:
        lebaran_dt = pd.to_datetime(lebaran_date).normalize()
        for i, dt in enumerate(d):
            if dt >= lebaran_dt - pd.Timedelta(days=30) and dt < lebaran_dt:
                is_pre_lebaran_30d[i] = 1
            elif dt == lebaran_dt:
                is_lebaran[i] = 1
            elif dt > lebaran_dt and dt <= lebaran_dt + pd.Timedelta(days=14):
                is_post_lebaran_14d[i] = 1

    return pd.DataFrame({
        "is_weekend": is_weekend,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "is_holiday": is_holiday,
        "is_pre_lebaran_30d": is_pre_lebaran_30d,
        "is_lebaran": is_lebaran,
        "is_post_lebaran_14d": is_post_lebaran_14d,
    })


def build_future_calendar_features(
    start_date,
    horizon: int,
    holiday_dates=None,
    lebaran_date=None,
):
    """Build calendar features for future dates."""
    start_date = pd.to_datetime(start_date)
    dates = pd.date_range(start=start_date, periods=horizon, freq="D")

    dow = dates.dayofweek
    is_weekend = (dow >= 5).astype(int)

    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    # Holiday features
    holiday_set = set(pd.to_datetime(list(holiday_dates)).normalize()) if holiday_dates is not None else set()
    is_holiday = pd.to_datetime(dates).normalize().isin(holiday_set).astype(int)

    # Lebaran features
    is_pre_lebaran_30d = np.zeros(horizon, dtype=int)
    is_lebaran = np.zeros(horizon, dtype=int)
    is_post_lebaran_14d = np.zeros(horizon, dtype=int)

    if lebaran_date:
        lebaran_dt = pd.to_datetime(lebaran_date).normalize()
        for i, d in enumerate(dates.normalize()):
            if d >= lebaran_dt - pd.Timedelta(days=30) and d < lebaran_dt:
                is_pre_lebaran_30d[i] = 1
            elif d == lebaran_dt:
                is_lebaran[i] = 1
            elif d > lebaran_dt and d <= lebaran_dt + pd.Timedelta(days=14):
                is_post_lebaran_14d[i] = 1

    future_exog = pd.DataFrame({
        "is_weekend": is_weekend,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "is_holiday": is_holiday,
        "is_pre_lebaran_30d": is_pre_lebaran_30d,
        "is_lebaran": is_lebaran,
        "is_post_lebaran_14d": is_post_lebaran_14d,
    })

    return future_exog


def compute_calendar_multipliers_from_g(train_g):
    """Compute calendar adjustment multipliers from training data."""
    eps = 1e-9
    tg = train_g.copy()

    base_mask = (
        (tg["is_weekend"] == 0) &
        (tg["is_pre_lebaran_30d"] == 0) &
        (tg["is_post_lebaran_14d"] == 0)
    )

    base = tg.loc[base_mask, "qty"].astype(float)
    base_mean = float(base.mean()) if len(base) > 0 else float(tg["qty"].astype(float).mean())
    base_mean = max(base_mean, eps)

    def ratio(mask, min_count=2):
        s = tg.loc[mask, "qty"].astype(float)
        if len(s) < min_count:
            return 1.0
        return float(s.mean() / base_mean)

    mult = {
        "weekend": ratio(tg["is_weekend"] == 1),
        "pre_lebaran": ratio(tg["is_pre_lebaran_30d"] == 1, min_count=1),
        "post_lebaran": ratio(tg["is_post_lebaran_14d"] == 1, min_count=1),
    }

    # Clamp multipliers
    for k in mult:
        mult[k] = float(np.clip(mult[k], 0.5, 2.5))

    return mult


def apply_calendar_adjustment_to_test(base_forecast, test_g, mult):
    """Apply calendar adjustments to test forecast."""
    base_forecast = np.asarray(base_forecast, dtype=float)
    out = []
    for i in range(len(test_g)):
        m = 1.0
        row = test_g.iloc[i]
        if int(row["is_weekend"]) == 1: 
            m *= mult["weekend"]
        if int(row["is_pre_lebaran_30d"]) == 1: 
            m *= mult["pre_lebaran"]
        if int(row["is_post_lebaran_14d"]) == 1: 
            m *= mult["post_lebaran"]
        out.append(base_forecast[i] * m)
    return np.asarray(out, dtype=float)


def apply_calendar_adjustment(base_forecast, future_flags, mult):
    """Apply calendar adjustments to future forecast."""
    adj = []
    for i, yhat in enumerate(np.asarray(base_forecast, dtype=float)):
        m = 1.0
        if future_flags.loc[i, "is_weekend"] == 1:
            m *= mult["weekend"]
        if future_flags.loc[i, "is_pre_lebaran_30d"] == 1:
            m *= mult["pre_lebaran"]
        if future_flags.loc[i, "is_post_lebaran_14d"] == 1:
            m *= mult["post_lebaran"]
        adj.append(yhat * m)
    return np.asarray(adj, dtype=float)


def prepare_sarimax_data(df, product_id, exog_cols):
    """Prepare data for SARIMAX model."""
    g = df[df["product_id"] == product_id].copy()
    g = g.sort_values("date")

    y = g["qty"].astype(float)
    X = g[exog_cols].astype(float)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=exog_cols,
        index=g.index
    )

    return y, X_scaled, scaler


def prepare_sarimax_product_df(
    df_ts,
    product_id,
    product_col="product_id",
    date_col="date",
    qty_col="qty",
    exog_cols=("is_weekend", "is_holiday", "dow_sin", "dow_cos", "is_pre_lebaran_30d", "is_lebaran", "is_post_lebaran_14d")
):
    """Prepare product dataframe for SARIMAX."""
    df = df_ts.copy()
    df[product_col] = df[product_col].astype(str).str.strip()
    df[date_col] = pd.to_datetime(df[date_col])

    g = df[df[product_col] == str(product_id)].sort_values(date_col).reset_index(drop=True)

    # Target
    y = g[qty_col].astype(float).to_numpy()

    # Exogenous
    X = g[list(exog_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    return y, X_scaled, scaler, list(X.columns), g


def analyze_sales_patterns(df_ts, product_id):
    """Analyze sales patterns for business insights."""
    g = df_ts[df_ts["product_id"] == product_id].copy()
    g = g.sort_values("date")
    
    if len(g) < 7:
        return {
            "trend": "insufficient_data",
            "volatility": "unknown",
            "weekend_effect": None,
            "growth_rate": None,
            "seasonality": "unknown",
            "avg_daily_sales": 0.0,
            "coefficient_of_variation": 0.0
        }
    
    sales = g["qty"].astype(float).to_numpy()
    
    # 1. Trend Analysis
    x = np.arange(len(sales))
    slope, _ = np.polyfit(x, sales, 1)
    
    if slope > 0.5:
        trend = "growing"
    elif slope < -0.5:
        trend = "declining"
    else:
        trend = "stable"
    
    # 2. Volatility (Coefficient of Variation)
    cv = np.std(sales) / (np.mean(sales) + 1e-9)
    if cv > 0.5:
        volatility = "high"
    elif cv > 0.3:
        volatility = "medium"
    else:
        volatility = "low"
    
    # 3. Weekend Effect
    weekend_avg = g[g["is_weekend"] == 1]["qty"].mean() if "is_weekend" in g.columns else None
    weekday_avg = g[g["is_weekend"] == 0]["qty"].mean() if "is_weekend" in g.columns else None
    
    if pd.notna(weekend_avg) and pd.notna(weekday_avg) and weekday_avg > 0:
        weekend_ratio = weekend_avg / weekday_avg
        if weekend_ratio > 1.3:
            weekend_effect = "strong_increase"
        elif weekend_ratio < 0.7:
            weekend_effect = "strong_decrease"
        else:
            weekend_effect = "neutral"
    else:
        weekend_effect = None
    
    # 4. Growth Rate (last 7 days vs previous 7 days)
    if len(sales) >= 14:
        recent_avg = np.mean(sales[-7:])
        previous_avg = np.mean(sales[-14:-7])
        growth_rate = ((recent_avg - previous_avg) / (previous_avg + 1e-9)) * 100
    else:
        growth_rate = None
    
    # 5. Seasonality Check
    if len(sales) >= 21:
        weekly_pattern = sales.reshape(-1, 7) if len(sales) % 7 == 0 else None
        if weekly_pattern is not None:
            weekly_std = np.std(np.mean(weekly_pattern, axis=0))
            seasonality = "detected" if weekly_std > np.mean(sales) * 0.2 else "weak"
        else:
            seasonality = "unknown"
    else:
        seasonality = "unknown"
    
    return {
        "trend": trend,
        "volatility": volatility,
        "weekend_effect": weekend_effect,
        "growth_rate": growth_rate,
        "seasonality": seasonality,
        "avg_daily_sales": float(np.mean(sales)),
        "coefficient_of_variation": float(cv)
    }
