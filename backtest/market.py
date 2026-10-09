"""Market-wide universe: every currently listed US common stock (Yahoo prices).

The universe is *not* defined by index membership. At each date the tradable set
is whatever was liquid at that time (price >= $5, top-N by 20-day dollar volume),
so a stock can be bought before it ever joins an index.

Still survivor-only: companies that were delisted before today are not on Yahoo.

    python -m backtest.market            # download (resumable)
"""
from __future__ import annotations

import io
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR, FIELDS, Panel

MARKET_DIR = DATA_DIR / "market"
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

_KEEP = re.compile(r"common stock|common shares|ordinary shares|class [a-z] (common|ordinary)|capital stock", re.I)
_DROP = re.compile(r"warrant|\brights?\b|\bunits?\b|preferred|\bpref\b|notes|debenture|\betf\b|\betn\b|\bfund\b|depositary|%|acquisition corp|subordinated", re.I)


def _read_list(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        txt = r.read().decode("utf-8", "ignore")
    lines = [l for l in txt.splitlines() if not l.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str).fillna("")


def filter_common_stocks(df: pd.DataFrame, sym_col: str, name_col: str = "Security Name") -> list[str]:
    """Keep plain common stock lines; drop ETFs, test issues, warrants, units, preferreds..."""
    d = df[(df.get("ETF", "N") == "N") & (df.get("Test Issue", "N") == "N")]
    ok = d[name_col].map(lambda n: bool(_KEEP.search(n)) and not _DROP.search(n))
    syms = d.loc[ok, sym_col].str.strip().str.upper()
    syms = syms[(syms != "") & ~syms.str.contains(r"[\$\^\s]")]
    return sorted({s.replace(".", "-").replace("/", "-") for s in syms})


def listed_common_stocks() -> list[str]:
    nas = filter_common_stocks(_read_list(NASDAQ_URL), "Symbol")
    oth = filter_common_stocks(_read_list(OTHER_URL), "ACT Symbol")
    return sorted(set(nas) | set(oth))


def download_market(tickers: list[str], start: str = "2003-06-01", batch: int = 100,
                    out_dir: Path = MARKET_DIR) -> None:
    """Download in batches, one parquet per batch; finished batches are skipped."""
    import yfinance as yf

    out_dir.mkdir(parents=True, exist_ok=True)
    n_batches = (len(tickers) + batch - 1) // batch
    for b in range(n_batches):
        f = out_dir / f"batch_{b:03d}.parquet"
        if f.exists():
            continue
        chunk = tickers[b * batch:(b + 1) * batch]
        raw = None
        for attempt in range(5):
            try:
                raw = yf.download(chunk, start=start, auto_adjust=True, progress=False,
                                  group_by="ticker", threads=True)
                break
            except Exception as exc:
                print(f"  batch {b}: {exc!r}; retry {attempt + 1}", flush=True)
                time.sleep(5 * (attempt + 1))
        frames = []
        if raw is not None and len(raw):
            have = set(raw.columns.get_level_values(0))
            for t in chunk:
                if t not in have:
                    continue
                d = raw[t].dropna(how="all")
                if d.empty:
                    continue
                d = d.rename(columns=str.lower)[FIELDS].reset_index()
                d.columns = ["date"] + FIELDS
                d["ticker"] = t
                frames.append(d)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker", *FIELDS])
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        for c in FIELDS:
            df[c] = df[c].astype("float32")
        df.to_parquet(f, index=False)
        print(f"batch {b + 1}/{n_batches}: {df['ticker'].nunique()} tickers with data", flush=True)


def load_long(out_dir: Path = MARKET_DIR, columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted(out_dir.glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data in {out_dir}; run python -m backtest.market")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def dollar_volume_rank_mask(close: pd.DataFrame, volume: pd.DataFrame, n: int = 500,
                            window: int = 20, min_price: float = 5.0) -> pd.DataFrame:
    """True where a stock is among the top-n by trailing average dollar volume that day."""
    dv = (close * volume).rolling(window, min_periods=window).mean()
    dv = dv.where(close >= min_price)
    return dv.rank(axis=1, ascending=False, method="first") <= n


def build_market_panel(n: int = 500, window: int = 20, min_price: float = 5.0,
                       out_dir: Path = MARKET_DIR) -> tuple[Panel, pd.DataFrame]:
    """Returns (panel of every stock that was ever in the top-n, bool mask dates x tickers)."""
    # pass 1: close & volume only, to find who was ever liquid enough
    px = load_long(out_dir, ["date", "ticker", "close", "volume"])
    px = px[(px["close"] > 0)]
    close = px.pivot(index="date", columns="ticker", values="close").sort_index()
    volume = px.pivot(index="date", columns="ticker", values="volume").reindex(close.index)
    mask = dollar_volume_rank_mask(close.astype("float64"), volume.astype("float64"), n, window, min_price)
    keep = sorted(mask.columns[mask.any()])
    del px, close, volume
    # pass 2: full OHLCV for the kept names
    df = load_long(out_dir)
    df = df[df["ticker"].isin(keep) & (df["close"] > 0) & (df["open"] > 0)]
    panel = Panel.from_long(df)
    mask = mask.reindex(index=panel.dates, columns=panel.tickers).fillna(False)
    return panel, mask


if __name__ == "__main__":
    tk = listed_common_stocks()
    print(len(tk), "listed common stocks")
    download_market(tk)
