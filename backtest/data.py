"""Download, cache and load daily OHLCV data as wide panels (dates x tickers)."""
from __future__ import annotations

import io
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA_DIR / "prices.parquet"
FIELDS = ["open", "high", "low", "close", "volume"]


@dataclass
class Panel:
    """Wide OHLCV panels sharing one date index and one ticker column order."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def subset(self, tickers: list[str]) -> "Panel":
        return Panel(*(getattr(self, f)[tickers] for f in FIELDS))

    @classmethod
    def from_long(cls, df: pd.DataFrame) -> "Panel":
        wide = {
            f: df.pivot(index="date", columns="ticker", values=f).sort_index()
            for f in FIELDS
        }
        cols = sorted(wide["close"].columns)
        return cls(*(wide[f][cols].astype("float64") for f in FIELDS))


def _download_yf(tickers: list[str], start: str, batch: int = 50) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        for attempt in range(4):
            try:
                raw = yf.download(
                    chunk, start=start, auto_adjust=True, progress=False,
                    group_by="ticker", threads=True,
                )
                break
            except Exception as exc:  # network hiccup: back off and retry
                print(f"  batch {i}: {exc!r}, retry {attempt + 1}")
                time.sleep(2 ** (attempt + 1))
        else:
            continue
        for t in chunk:
            if t not in raw.columns.get_level_values(0):
                continue
            sub = raw[t].dropna(how="all")
            if sub.empty:
                continue
            sub = sub.rename(columns=str.lower)[FIELDS].reset_index()
            sub = sub.rename(columns={sub.columns[0]: "date"})
            sub["ticker"] = t
            frames.append(sub)
        print(f"  downloaded {min(i + batch, len(tickers))}/{len(tickers)}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _download_stooq(ticker: str, start: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        df = pd.read_csv(io.StringIO(resp.read().decode()))
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= start]
    df["ticker"] = ticker
    return df[["date", "ticker", *FIELDS]]


def download(tickers: list[str], start: str = "2004-01-01") -> pd.DataFrame:
    df = _download_yf(tickers, start)
    got = set(df["ticker"]) if not df.empty else set()
    missing = [t for t in tickers if t not in got]
    if missing:
        print(f"  yfinance missed {len(missing)} tickers, trying stooq")
        extra = []
        for t in missing:
            try:
                extra.append(_download_stooq(t, start))
            except Exception as exc:
                print(f"  stooq {t}: {exc!r}")
        if extra:
            df = pd.concat([df, *extra], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df[["date", "ticker", *FIELDS]]


def load_or_download(tickers: list[str], start: str = "2004-01-01",
                     refresh: bool = False) -> Panel:
    if CACHE.exists() and not refresh:
        df = pd.read_parquet(CACHE)
    else:
        DATA_DIR.mkdir(exist_ok=True)
        df = download(tickers, start)
        if df.empty:
            raise RuntimeError("no data downloaded - check network access")
        df.to_parquet(CACHE, index=False)
    df = df.dropna(subset=["close"])
    df = df[(df["close"] > 0) & (df["open"] > 0)]
    return Panel.from_long(df)


def synthetic_panel(n_tickers: int = 30, n_days: int = 2500, seed: int = 0,
                    start: str = "2010-01-01") -> Panel:
    """Random-walk OHLCV with mild mean reversion; used for tests/offline demo."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    drift = 0.0003
    rets = rng.normal(drift, 0.018, (n_days, n_tickers))
    rets[1:] -= 0.05 * rets[:-1]
    close = 50 * np.exp(np.cumsum(rets, axis=0))
    gap = rng.normal(0, 0.006, (n_days, n_tickers))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(gap)
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.008, close.shape)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.008, close.shape)))
    vol = rng.lognormal(14, 0.5, close.shape)
    mk = lambda a: pd.DataFrame(a, index=dates, columns=tickers)
    return Panel(mk(open_), mk(hi), mk(lo), mk(close), mk(vol))
