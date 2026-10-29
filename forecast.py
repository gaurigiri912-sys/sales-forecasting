"""
AI Sales Forecasting - core pipeline
Steps: load data -> monthly series -> features -> train/test (time split)
       -> evaluate models -> forecast future months
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FEATURES = ["month", "lag1", "lag2", "lag3", "lag12", "roll3"]


# ---------------------------------------------------------------
# 1. LOAD DATA (Kaggle Superstore CSV, or synthetic demo data)
# ---------------------------------------------------------------
def make_demo_data():
    """Demo data so the project runs even without a CSV file."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    regions = ["East", "West", "North", "South"]
    categories = ["Furniture", "Technology", "Office Supplies"]
    rows = []
    for d in dates:
        trend = 1 + 0.0004 * (d - dates[0]).days
        season = 1 + 0.3 * np.sin(2 * np.pi * (d.month - 3) / 12)
        if d.month in (11, 12):
            season += 0.3
        for r in regions:
            for c in categories:
                base = 180 * trend * season
                rows.append((d, r, c, max(base + rng.normal(0, 40), 5)))
    return pd.DataFrame(rows, columns=["date", "region", "category", "sales"])


def load_data(path="data/sales.csv"):
    """Superstore columns: Order Date, Sales, Region, Category."""
    base = os.path.dirname(os.path.abspath(__file__))
    # look for the CSV in data/ or right next to the code files
    candidates = [path, "sales.csv",
                  os.path.join(base, "data", "sales.csv"),
                  os.path.join(base, "sales.csv")]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        print("WARNING: sales.csv not found - using demo data")
        return make_demo_data()
    print(f"Loading data from: {path}")
    df = pd.read_csv(path, encoding="latin-1")
    df = df.rename(columns={
        "Order Date": "date", "Sales": "sales",
        "Region": "region", "Category": "category",
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.drop_duplicates().dropna(subset=["date", "sales"])
    for col in ("region", "category"):
        if col not in df.columns:
            df[col] = "All"
    return df


def monthly_series(df):
    """Total sales per month."""
    return df.set_index("date")["sales"].resample("MS").sum()


# ---------------------------------------------------------------
# 2. FEATURES
# ---------------------------------------------------------------
def row_features(date, hist):
    """Features for one month, using only past values (hist)."""
    return {
        "month": date.month,
        "lag1": hist[-1],
        "lag2": hist[-2],
        "lag3": hist[-3],
        "lag12": hist[-12],
        "roll3": float(np.mean(hist[-3:])),
    }


def build_table(s):
    vals = s.values.astype(float)
    rows = []
    for i in range(12, len(s)):
        r = row_features(s.index[i], list(vals[:i]))
        r["y"] = vals[i]
        r["date"] = s.index[i]
        rows.append(r)
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------
# 3. METRICS
# ---------------------------------------------------------------
def get_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAPE %": float(np.mean(np.abs((y - p) / y)) * 100),
    }


def get_models():
    models = {"Random Forest": RandomForestRegressor(n_estimators=300, random_state=42)}
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        )
    return models


# ---------------------------------------------------------------
# 4. TRAIN, EVALUATE, FORECAST
# ---------------------------------------------------------------
def run_pipeline(s, horizon=6, test_months=6):
    table = build_table(s)
    if len(table) < test_months + 6:
        raise ValueError("Not enough data. Need at least ~2 years of monthly data.")

    # time-based split (never shuffle time series)
    train, test = table.iloc[:-test_months], table.iloc[-test_months:]

    results, preds = [], {}

    # baseline: same month last year
    preds["Seasonal Naive"] = test["lag12"].values
    results.append({"Model": "Seasonal Naive", **get_metrics(test["y"], preds["Seasonal Naive"])})

    ml_scores = {}
    for name, model in get_models().items():
        model.fit(train[FEATURES], train["y"])
        p = model.predict(test[FEATURES])
        preds[name] = p
        m = get_metrics(test["y"], p)
        ml_scores[name] = m["RMSE"]
        results.append({"Model": name, **m})

    metrics = pd.DataFrame(results).round(2)
    best = min(ml_scores, key=ml_scores.get)  # lowest RMSE

    # refit best model on ALL data, then forecast future months
    final_model = get_models()[best]
    final_model.fit(table[FEATURES], table["y"])

    hist = list(s.values.astype(float))
    dates = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    future = []
    for d in dates:
        x = pd.DataFrame([row_features(d, hist)])[FEATURES]
        yhat = float(final_model.predict(x)[0])
        future.append(yhat)
        hist.append(yhat)

    return {
        "metrics": metrics,
        "best": best,
        "model": final_model,
        "test": pd.DataFrame({"actual": test["y"], "predicted": preds[best]}),
        "future": pd.Series(future, index=dates, name="forecast"),
    }


if __name__ == "__main__":
    df = load_data()
    s = monthly_series(df)
    out = run_pipeline(s, horizon=6)

    print("\n=== Model comparison (lower is better) ===")
    print(out["metrics"].to_string(index=False))
    print(f"\nBest model: {out['best']}")
    print("\n=== Next 6 months forecast ===")
    print(out["future"].round(0).to_string())

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(out["model"], "outputs/best_model.pkl")
    out["future"].to_csv("outputs/forecast.csv")
    print("\nSaved: outputs/best_model.pkl, outputs/forecast.csv")
