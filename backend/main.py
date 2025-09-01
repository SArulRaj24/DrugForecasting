from __future__ import annotations

import os
import io
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database.db_manager import init_db, upsert_monthly_records, fetch_categories, fetch_monthly_records, fetch_series
from utils.data_processor import parse_upload_to_monthly_long, pivot_wide, summarize_wide
from utils.model_loader import load_model


# App / Config
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sales.db")
app = FastAPI(title="Medicine Sales Forecast API", version="1.0.0")

# CORS for dev — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB (creates tables if needed)
init_db(DATABASE_URL)

# Schemas
class PredictRequest(BaseModel):
    category: str = Field(..., description="ATC category to forecast (e.g., M01AB)")
    periods: int = Field(..., ge=1, le=36, description="Forecast horizon in months")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)


# ----------------------
# Endpoints
# ----------------------
@app.get("/api/health")
async def health() -> Dict[str, Any]:
    try:
        cats = fetch_categories()
        status = "connected"
    except Exception:
        status = "disconnected"
    return {"status": "ok", "database": status, "time": datetime.utcnow().isoformat() + "Z"}


@app.get("/api/categories")
def categories() -> Dict[str, List[str]]:
    return {"categories": fetch_categories()}


@app.get("/api/data")
def get_data(start_date: Optional[str] = None, end_date: Optional[str] = None, categories: Optional[str] = None) -> Dict[str, Any]:
    rows = fetch_monthly_records(start_date=start_date, end_date=end_date, categories=categories)
    if not rows:
        raise HTTPException(status_code=404, detail="No data found for the given filters.")

    df = pd.DataFrame(rows, columns=["date", "category", "quantity"])
    df["date"] = pd.to_datetime(df["date"])

    wide = pivot_wide(df)
    summary = summarize_wide(wide)

    wide_out = wide.copy()
    wide_out["date"] = wide_out["date"].dt.strftime("%Y-%m-%d")

    return {"data": wide_out.to_dict(orient="records"), "summary_stats": summary, "categories": fetch_categories()}

@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    monthly_long = parse_upload_to_monthly_long(filename=file.filename or "", content=content)
    inserted = upsert_monthly_records(monthly_long)
    cats = fetch_categories()
    return {"message": f"Ingested {inserted} monthly records across {monthly_long['category'].nunique()} categories.", "categories": cats}


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        model, transformer, model_type = load_model(req.category)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    # Fetch historical data
    y = fetch_series(req.category)
    if y.empty:
        raise HTTPException(status_code=404, detail=f"No historical data found for category {req.category}")

    last_date = y.index[-1]
    history_len = len(y)

    if model_type == "prophet":
        # Prophet: predict only future
        future = model.make_future_dataframe(periods=req.periods, freq="M")
        forecast = model.predict(future)

        # Take only future predictions
        future_forecast = forecast.tail(req.periods)
        dates = future_forecast["ds"].dt.strftime("%Y-%m-%d").tolist()
        preds = future_forecast["yhat"].tolist()

    elif model_type in ["arima", "sarima"]:
    # ARIMA/SARIMA: predict last 12 months + future
        start_idx = max(0, history_len - 12)   # go back exactly 1 year
        end_idx = history_len + req.periods - 1

        preds_obj = model.get_prediction(start=start_idx, end=end_idx)
        preds_mean = preds_obj.predicted_mean

        if transformer is not None:
            preds = transformer.inverse_transform(
                preds_mean.to_numpy().reshape(-1, 1)
            ).flatten()
        else:
            preds = preds_mean.to_numpy().flatten()

        # Dates: last 12 months + future
        past_dates = y.index[-12:]   # last 12 months
        future_dates = pd.date_range(
            last_date + pd.offsets.MonthBegin(1),
            periods=req.periods,
            freq="MS"
        )
        dates = [d.strftime("%Y-%m-%d") for d in past_dates.tolist() + future_dates.tolist()]

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

    return {
        "category": req.category,
        "dates": dates,
        "predictions": [float(x) for x in preds]
    }




@app.get("/api/stats/{category}")
async def stats(category: str) -> Dict[str, Any]:
    y = fetch_series(category)
    if y.empty:
        raise HTTPException(status_code=404, detail=f"No data found for category {category}")
    df = y.to_frame(name="quantity")
    return {
        "category": category,
        "record_count": int(len(df)),
        "start_date": df.index.min().strftime("%Y-%m-%d"),
        "end_date": df.index.max().strftime("%Y-%m-%d"),
        "avg_quantity": float(df["quantity"].mean()),
        "total_quantity": float(df["quantity"].sum()),
        "min_quantity": float(df["quantity"].min()),
        "max_quantity": float(df["quantity"].max()),
    }
