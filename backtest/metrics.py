"""Trade-level and equity-curve statistics."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

OOS_START = pd.Timestamp("2019-01-01")


def trade_stats(trades: pd.DataFrame, prefix: str = "") -> dict:
    r = trades["ret"].to_numpy() if len(trades) else np.array([])
    n = len(r)
    if n == 0:
        return {f"{prefix}trades": 0}
    wins, losses = r[r > 0], r[r <= 0]
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    gross_loss = -losses.sum()
    sd = r.std(ddof=1) if n > 1 else np.nan
    srt = np.sort(r)[::-1]
    k5 = max(1, int(n * 0.05))
    return {
        f"{prefix}trades": n,
        f"{prefix}win_rate": len(wins) / n,
        f"{prefix}avg_ret": r.mean(),
        f"{prefix}median_ret": float(np.median(r)),
        f"{prefix}avg_win": avg_win,
        f"{prefix}avg_loss": avg_loss,
        f"{prefix}payoff": avg_win / -avg_loss if avg_loss < 0 else np.inf,
        f"{prefix}profit_factor": wins.sum() / gross_loss if gross_loss > 0 else np.inf,
        f"{prefix}worst": r.min(),
        f"{prefix}t_stat": r.mean() / (sd / math.sqrt(n)) if n > 1 and sd > 0 else np.nan,
        f"{prefix}avg_hold": trades["hold"].mean(),
        f"{prefix}big_win_rate": float((r > 0.5).mean()),   # share of trades > +50%
        f"{prefix}best": float(r.max()),
        f"{prefix}top5_share": float(srt[:k5].sum() / r.sum()) if r.sum() > 0 else np.nan,
    }


def yearly_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["year", "trades", "win_rate", "avg_ret"])
    y = trades["entry_date"].dt.year
    g = trades.groupby(y)["ret"]
    return pd.DataFrame({
        "trades": g.size(), "win_rate": g.apply(lambda s: (s > 0).mean()),
        "avg_ret": g.mean(),
    }).rename_axis("year").reset_index()


def equity_stats(eq: pd.Series, prefix: str = "pf_") -> dict:
    eq = eq.dropna()
    eq = eq[eq.index >= eq.index[0]]
    if len(eq) < 2:
        return {}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dr = eq.pct_change().dropna()
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    dd = eq / eq.cummax() - 1
    sharpe = dr.mean() / dr.std() * math.sqrt(252) if dr.std() > 0 else np.nan
    # calendar-year returns (first year measured from the first observation)
    ye = eq.resample("YE").last()
    prev = ye.shift(1)
    prev.iloc[0] = eq.iloc[0]
    yr = ye / prev - 1
    return {
        f"{prefix}final": eq.iloc[-1] / eq.iloc[0],
        f"{prefix}worst_year": yr.min(),
        f"{prefix}worst_year_n": int(yr.idxmin().year),
        f"{prefix}best_year": yr.max(),
        f"{prefix}losing_years": int((yr < 0).sum()),
        f"{prefix}n_years": int(len(yr)),
        f"{prefix}cagr": cagr,
        f"{prefix}sharpe": sharpe,
        f"{prefix}max_dd": dd.min(),
        f"{prefix}vol": dr.std() * math.sqrt(252),
        f"{prefix}calmar": cagr / -dd.min() if dd.min() < 0 else np.nan,
    }


def full_stats(trades: pd.DataFrame, eq: pd.Series | None = None) -> dict:
    out = trade_stats(trades)
    is_ = trades[trades["entry_date"] < OOS_START] if len(trades) else trades
    oos = trades[trades["entry_date"] >= OOS_START] if len(trades) else trades
    out.update(trade_stats(is_, "is_"))
    out.update(trade_stats(oos, "oos_"))
    ys = yearly_stats(trades)
    ys = ys[ys["trades"] >= 20]
    if len(ys):
        out["years_positive"] = float((ys["avg_ret"] > 0).mean())
        out["yearly_win_rate_min"] = float(ys["win_rate"].min())
    if eq is not None:
        out.update(equity_stats(eq))
        out.update(equity_stats(eq[eq.index >= OOS_START], "pf_oos_"))
    return out
