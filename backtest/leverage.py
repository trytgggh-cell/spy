"""Index trend / mean-reversion with synthetic leveraged QQQ (rules: docs/V3_PLAN.md).

All simulation is close-to-close with a one-day execution lag: a signal seen at
the close of day t is traded at the close of day t+1 (conservative proxy for the
next open). Cash earns the 13-week T-bill yield.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .data import DATA_DIR

CACHE = DATA_DIR / "underlying.parquet"


def load_underlying(start: str = "1998-06-01", refresh: bool = False) -> pd.DataFrame:
    """Adjusted closes of QQQ, SPY, QLD, TQQQ and the 13-week yield ^IRX (percent)."""
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    import yfinance as yf

    raw = yf.download(["QQQ", "SPY", "QLD", "TQQQ", "^IRX"], start=start, auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    out = pd.DataFrame({t: raw[t]["Close"] for t in ["QQQ", "SPY", "QLD", "TQQQ", "^IRX"]})
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out = out.sort_index()
    DATA_DIR.mkdir(exist_ok=True)
    out.to_parquet(CACHE)
    return out


def daily_rf(irx: pd.Series) -> pd.Series:
    return (irx.ffill().bfill() / 100.0 / 252.0).rename("rf")


def synth_leverage(r: pd.Series, rf: pd.Series, L: int, spread: float = 0.005,
                   expense: float = 0.0095) -> pd.Series:
    """Daily-rebalanced L-times return, net of financing and fund expenses."""
    if L == 1:
        return r
    return L * r - (L - 1) * (rf + spread / 252.0) - expense / 252.0


def _trades_from_mask(held: np.ndarray, ret: np.ndarray, dates: pd.DatetimeIndex, cost: float):
    rows = []
    n = len(held)
    d = np.diff(np.concatenate([[False], held, [False]]).astype(int))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):      # held[a:b]
        g = np.prod(1 + ret[a:b]) * (1 - cost) / (1 + cost) - 1
        rows.append({"ticker": "QQQ", "entry_idx": a - 1, "exit_idx": b - 1,
                     "entry_date": dates[max(a - 1, 0)], "exit_date": dates[min(b - 1, n - 1)],
                     "ret": g, "hold": b - a})
    return pd.DataFrame(rows)


def run_trend(sig: pd.Series, r_lev: pd.Series, rf: pd.Series, cost: float):
    """Hold the leveraged series while `sig` (decided at the close) is True."""
    held = sig.shift(2).fillna(False).to_numpy(bool)       # signal t -> trade close t+1 -> earns day t+2
    ret = r_lev.fillna(0.0).to_numpy()
    cash = rf.reindex(sig.index).fillna(0.0).to_numpy()
    switch = np.abs(np.diff(held.astype(int), prepend=0))
    day = np.where(held, ret, cash) - switch * cost
    eq = pd.Series(np.cumprod(1 + day), index=sig.index)
    return eq, _trades_from_mask(held, ret, sig.index, cost)


def run_reversion(entry: pd.Series, exit_: pd.Series, r_lev: pd.Series, rf: pd.Series,
                  cost: float, max_hold: int = 10):
    """Enter at close t+1 after an entry signal at t; leave at close j+1 once exit_[j]
    (or max_hold days) is reached."""
    n = len(entry)
    en, ex = entry.to_numpy(bool), exit_.to_numpy(bool)
    held = np.zeros(n, bool)
    i = 0
    while i < n - 2:
        if not en[i]:
            i += 1
            continue
        e = i + 1
        j = e
        while j < n - 1 and not (ex[j] or (j - e) >= max_hold):
            j += 1
        x = min(j + 1, n - 1)
        held[e + 1: x + 1] = True
        i = x
    ret = r_lev.fillna(0.0).to_numpy()
    cash = rf.reindex(entry.index).fillna(0.0).to_numpy()
    switch = np.abs(np.diff(held.astype(int), prepend=0))
    day = np.where(held, ret, cash) - switch * cost
    eq = pd.Series(np.cumprod(1 + day), index=entry.index)
    return eq, _trades_from_mask(held, ret, entry.index, cost)


def signals(price: pd.Series):
    c = price.to_frame("p")
    sma = lambda n: ind.sma(c, n)["p"]
    return {
        "T-100": price > sma(100), "T-150": price > sma(150), "T-200": price > sma(200),
        "T-X": sma(50) > sma(200),
        "_up200": price > sma(200),
        "_rsi2": ind.rsi(c, 2)["p"], "_above5": price > sma(5),
    }
