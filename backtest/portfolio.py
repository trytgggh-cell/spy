"""Portfolio-level simulations.

* simulate_slots: take a trade list and run it through an account with at most
  N concurrent positions, each sized at equity/N on entry. Positions are marked
  to market at the close every day, so drawdowns are realistic.
* momentum_rotation: monthly top-N momentum portfolio.
"""
from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd

from .data import Panel


@nb.njit(cache=True)
def _slots(close, col, ei, xi, ep, xp, max_pos):
    n, m = close.shape
    last_px = np.full(m, np.nan)
    pos_col = np.full(max_pos, -1, np.int64)
    pos_xi = np.zeros(max_pos, np.int64)
    pos_sh = np.zeros(max_pos)
    pos_xp = np.zeros(max_pos)
    holding = np.zeros(m, np.bool_)
    accepted = np.zeros(col.size, np.bool_)
    equity = np.empty(n)
    cash = 1.0
    prev_eq = 1.0
    k = 0
    for d in range(n):
        for j in range(m):
            if not np.isnan(close[d, j]):
                last_px[j] = close[d, j]
        # exits scheduled today (sold at open or at the intraday stop price)
        for s in range(max_pos):
            if pos_col[s] >= 0 and pos_xi[s] == d:
                cash += pos_sh[s] * pos_xp[s]
                holding[pos_col[s]] = False
                pos_col[s] = -1
        # entries at today's open, sized off yesterday's equity
        slot_cash = prev_eq / max_pos
        while k < col.size and ei[k] < d:
            k += 1
        while k < col.size and ei[k] == d:
            c = col[k]
            free = -1
            for s in range(max_pos):
                if pos_col[s] < 0:
                    free = s
                    break
            alloc = min(slot_cash, cash)
            if free >= 0 and not holding[c] and alloc > 1e-9:
                sh = alloc / ep[k]
                cash -= alloc
                accepted[k] = True
                if xi[k] == d:  # same-day stop-out
                    cash += sh * xp[k]
                else:
                    pos_col[free] = c
                    pos_xi[free] = xi[k]
                    pos_sh[free] = sh
                    pos_xp[free] = xp[k]
                    holding[c] = True
            k += 1
        mtm = 0.0
        for s in range(max_pos):
            if pos_col[s] >= 0:
                mtm += pos_sh[s] * last_px[pos_col[s]]
        equity[d] = cash + mtm
        prev_eq = equity[d]
    return equity, accepted


def simulate_slots(panel: Panel, trades: pd.DataFrame, max_pos: int = 10,
                   seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """Returns (daily equity curve starting at 1.0, accepted trades)."""
    dates = panel.dates
    if trades.empty:
        return pd.Series(1.0, index=dates), trades
    # Priority among same-day candidates: lowest score first (e.g. lowest RSI),
    # otherwise random so there is no alphabetical bias.
    rng = np.random.default_rng(seed)
    prio = trades["score"].to_numpy() if "score" in trades else rng.random(len(trades))
    prio = np.where(np.isnan(prio), np.inf, prio)
    order = np.lexsort((prio, trades["entry_idx"].to_numpy()))
    tr = trades.iloc[order]
    col_of = {t: i for i, t in enumerate(panel.tickers)}
    col = tr["ticker"].map(col_of).to_numpy(np.int64)
    eq, acc = _slots(np.ascontiguousarray(panel.close.to_numpy(np.float64)), col,
                     tr["entry_idx"].to_numpy(np.int64), tr["exit_idx"].to_numpy(np.int64),
                     tr["entry_px"].to_numpy(np.float64), tr["exit_px"].to_numpy(np.float64),
                     max_pos)
    return pd.Series(eq, index=dates), tr[acc]


def momentum_rotation(panel: Panel, lookback: int = 252, skip: int = 21,
                      top_n: int = 10, market_filter: pd.Series | None = None,
                      cost: float = 0.0005, start_idx: int = 260,
                      eligible: pd.DataFrame | None = None):
    """Monthly rotation into the top-N momentum names.

    Signal at the last close of each month; trades at the next open.
    market_filter: bool Series (True = risk-on). When False -> hold cash.
    eligible: optional bool DataFrame (dates x tickers) of what is tradable that
    day (e.g. a market-wide liquidity ranking). It is a point-in-time universe
    rule, not an index-membership rule.
    Returns (equity Series, holdings DataFrame with one row per stock-month).
    """
    c = panel.close
    o = panel.open
    mom = c.shift(skip) / c.shift(lookback) - 1
    dates = panel.dates
    month = dates.to_period("M")
    month_end = np.flatnonzero(month[:-1] != month[1:])  # last day of each month
    month_end = month_end[month_end >= start_idx]
    eq = pd.Series(np.nan, index=dates)
    rows = []
    cur = 1.0
    for k, s in enumerate(month_end):
        e = s + 1
        x = month_end[k + 1] + 1 if k + 1 < len(month_end) else len(dates) - 1
        risk_on = True if market_filter is None else bool(market_filter.iloc[s])
        if not risk_on:
            eq.iloc[e : x + 1] = cur
            continue
        m = mom.iloc[s].dropna()
        # require tradable prices at entry
        m = m[o.iloc[e][m.index].notna()]
        if eligible is not None:
            m = m[eligible.iloc[s].reindex(m.index, fill_value=False).to_numpy(bool)]
        picks = m.nlargest(top_n).index
        if len(picks) == 0:
            eq.iloc[e : x + 1] = cur
            continue
        ep = o.iloc[e][picks] * (1 + cost)
        path = c.iloc[e : x + 1][picks].ffill() / ep
        xp = o.iloc[x][picks].fillna(c.iloc[x - 1][picks]) * (1 - cost)
        path.iloc[-1] = xp / ep  # last day valued at exit price
        port = path.mean(axis=1) * cur
        eq.iloc[e : x + 1] = port.to_numpy()
        cur = float(port.iloc[-1])
        for t in picks:
            rows.append({
                "ticker": t, "entry_idx": e, "exit_idx": x,
                "entry_date": dates[e], "exit_date": dates[x],
                "entry_px": ep[t], "exit_px": xp[t], "ret": xp[t] / ep[t] - 1,
                "hold": x - e, "reason": "rebalance",
            })
    eq = eq.ffill().fillna(1.0)
    return eq, pd.DataFrame(rows)
