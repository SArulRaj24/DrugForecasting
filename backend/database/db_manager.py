from __future__ import annotations
from typing import List, Tuple, Optional
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Date, Float, select, and_, delete
from sqlalchemy.engine import Engine

_engine: Engine = None
_metadata = MetaData()

sales_table = Table(
    "sales", _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("date", Date, nullable=False),  # month start
    Column("category", String, nullable=False, index=True),
    Column("quantity", Float, nullable=False),
)


def init_db(database_url: str) -> None:
    global _engine
    _engine = create_engine(database_url, future=True)
    _metadata.create_all(_engine)


def upsert_monthly_records(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    df["category"] = df["category"].astype(str)
    df["quantity"] = df["quantity"].astype(float)

    records = df.to_dict(orient="records")
    with _engine.begin() as conn:
        # delete existing (date, category) to emulate upsert
        unique_pairs = {(r["date"].date(), r["category"]) for r in records}
        for d, c in unique_pairs:
            conn.execute(delete(sales_table).where(and_(sales_table.c.date == d, sales_table.c.category == c)))
        conn.execute(sales_table.insert(), records)
    return len(records)


def fetch_categories() -> List[str]:
    with _engine.begin() as conn:
        rows = conn.execute(select(sales_table.c.category).distinct().order_by(sales_table.c.category)).all()
    return [r[0] for r in rows]


def fetch_monthly_records(start_date: Optional[str] = None, end_date: Optional[str] = None, categories: Optional[str] = None) -> List[Tuple]:
    stmt = select(sales_table.c.date, sales_table.c.category, sales_table.c.quantity)
    if start_date:
        dt = pd.to_datetime(start_date)
        stmt = stmt.where(sales_table.c.date >= dt.date())
    if end_date:
        dt = pd.to_datetime(end_date)
        stmt = stmt.where(sales_table.c.date <= dt.date())
    if categories:
        cats = [c.strip() for c in categories.split(",") if c.strip()]
        if cats:
            stmt = stmt.where(sales_table.c.category.in_(cats))

    with _engine.begin() as conn:
        rows = conn.execute(stmt.order_by(sales_table.c.date)).all()
    return rows


def fetch_series(category: str) -> pd.Series:
    stmt = select(sales_table.c.date, sales_table.c.quantity).where(sales_table.c.category == category).order_by(sales_table.c.date)
    with _engine.begin() as conn:
        rows = conn.execute(stmt).all()
    if not rows:
        raise ValueError(f"No data found for category '{category}'.")
    df = pd.DataFrame(rows, columns=["date", "quantity"]).dropna()
    df["date"] = pd.to_datetime(df["date"])  # ensure timestamp

    idx = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    s = df.set_index("date")["quantity"].reindex(idx, fill_value=0.0)
    s.index.name = "date"
    return s
