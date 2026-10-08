"""Vectorised indicators on wide DataFrames (dates x tickers)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder RSI."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn > 0, 100.0).where(up.notna())


def bollinger(close: pd.DataFrame, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    return mid - k * sd, mid, mid + k * sd


def true_range(high, low, close) -> pd.DataFrame:
    pc = close.shift(1)
    return np.maximum(high - low, np.maximum((high - pc).abs(), (low - pc).abs()))


def atr(high, low, close, n: int = 14) -> pd.DataFrame:
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def ibs(high, low, close) -> pd.DataFrame:
    """Internal bar strength: where the close sits in the day's range (0..1)."""
    rng = (high - low).replace(0, np.nan)
    return (close - low) / rng


def highest(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).max()


def lowest(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).min()


def macd(close: pd.DataFrame, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig


def down_streak(close: pd.DataFrame) -> pd.DataFrame:
    """Number of consecutive lower closes ending today."""
    down = (close.diff() < 0).to_numpy()
    out = np.zeros(down.shape, np.int32)
    for i in range(1, len(out)):
        out[i] = np.where(down[i], out[i - 1] + 1, 0)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def cross_above(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_below(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return (a < b) & (a.shift(1) >= b.shift(1))
