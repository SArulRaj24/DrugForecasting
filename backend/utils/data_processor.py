from __future__ import annotations
import io
import pandas as pd
from typing import Dict, Any


def parse_upload_to_monthly_long(filename: str, content: bytes) -> pd.DataFrame:
    name = (filename or "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError("Unsupported file type. Upload CSV or XLSX.")

    # Normalize date column: accept 'date' or 'datum'
    date_cols = [c for c in df.columns if c.lower() in ["date", "datum"]]
    if not date_cols:
        raise ValueError("File must contain a 'date' column.")
    df = df.rename(columns={date_cols[0]: "date"})

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # keep only numeric category columns
    ignore_cols = ["year", "month", "hour"]
    cat_cols = [c for c in df.columns if c != "date" and c.lower() not in ignore_cols]
    if not cat_cols:
        raise ValueError("No category columns found besides 'date'.")

    numeric_cols = []
    for c in cat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().any():
            numeric_cols.append(c)
    if not numeric_cols:
        raise ValueError("No numeric category columns found.")

    long_df = df.melt(id_vars=["date"], value_vars=numeric_cols, var_name="category", value_name="quantity")
    long_df["quantity"] = long_df["quantity"].fillna(0.0).astype(float).clip(lower=0.0)

    long_df["month"] = long_df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = long_df.groupby(["month", "category"], as_index=False)["quantity"].sum()
    monthly.rename(columns={"month": "date"}, inplace=True)
    return monthly


def pivot_wide(df_long: pd.DataFrame) -> pd.DataFrame:  
    wide = df_long.pivot(index="date", columns="category", values="quantity").sort_index()
    wide = wide.fillna(0.0)
    wide = wide.reset_index()
    return wide


def summarize_wide(wide: pd.DataFrame) -> Dict[str, Any]:
    import numpy as np
    summary: Dict[str, Any] = {}
    for c in [col for col in wide.columns if col != "date"]:
        s = wide[c]
        summary[c] = {
            "mean": float(np.nanmean(s)),
            "median": float(np.nanmedian(s)),
            "std": float(np.nanstd(s)),
            "min": float(np.nanmin(s)),
            "max": float(np.nanmax(s)),
            "total": float(np.nansum(s)),
            "count": int(np.count_nonzero(~np.isnan(s)))
        }
    return summary