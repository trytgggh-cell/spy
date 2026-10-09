"""A basket of liquid ETFs (equity, sector, bond, commodity).

ETFs are a cleaner test bed than single stocks: the basket does not lose its
failures to delisting, so the survivorship bias that affects stock backtests
is mostly absent.
"""
from __future__ import annotations

import pandas as pd

from .data import DATA_DIR, Panel, _download_yf

ETFS = {
    "equity": ["SPY", "QQQ", "IWM", "MDY", "EFA", "EEM", "VNQ"],
    "sector": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"],
    "bond": ["TLT", "IEF", "LQD", "HYG"],
    "commodity": ["GLD", "SLV", "DBC", "USO"],
}
ALL = [t for v in ETFS.values() for t in v]
SECTORS = ETFS["sector"]
CACHE = DATA_DIR / "etf.parquet"


def load_etf_panel(start: str = "2003-06-01", refresh: bool = False) -> Panel:
    if CACHE.exists() and not refresh:
        df = pd.read_parquet(CACHE)
    else:
        DATA_DIR.mkdir(exist_ok=True)
        df = _download_yf(ALL, start)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df.to_parquet(CACHE, index=False)
    df = df.dropna(subset=["close"])
    df = df[(df["close"] > 0) & (df["open"] > 0)]
    return Panel.from_long(df)
