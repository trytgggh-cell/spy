"""Per-ticker trade simulation (numba).

Rules (no look-ahead):
  * entry signal is evaluated on day i's close -> buy at open[i+1]
  * stop / target are checked intraday from the entry day on; stop wins ties
  * exit condition or max-hold is evaluated on day j's close -> sell at open[j+1]
  * one position per ticker; signals while in a position are ignored
  * cost is charged per side as a fraction of price (commission + slippage)
"""
from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np
import pandas as pd

from .data import Panel


@dataclass
class ExitRule:
    max_hold: int = 10                 # trading days, counted from entry day
    stop_pct: float = 0.0              # e.g. 0.08 = 8% hard stop, 0 = off
    target_pct: float = 0.0            # e.g. 0.10 = +10% take-profit, 0 = off
    trail_atr: float = 0.0             # chandelier stop: highest close - k*ATR
    # exit condition matrix is passed separately (bool DataFrame or None)


@nb.njit(cache=True)
def _simulate(o, h, l, c, entry, exit_sig, atr, max_hold, stop_pct, tgt_pct,
              trail_k, cost):
    n, m = c.shape
    cap = 1 << 16
    out_col = np.empty(cap, np.int32)
    out_ei = np.empty(cap, np.int32)
    out_xi = np.empty(cap, np.int32)
    out_ep = np.empty(cap, np.float64)
    out_xp = np.empty(cap, np.float64)
    out_reason = np.empty(cap, np.int8)  # 0 time,1 signal,2 stop,3 target,4 trail,5 data end
    k = 0
    for col in range(m):
        i = 0
        while i < n - 1:
            if not entry[i, col] or np.isnan(o[i + 1, col]):
                i += 1
                continue
            e = i + 1
            ep = o[e, col]
            stop = ep * (1 - stop_pct) if stop_pct > 0 else -1.0
            tgt = ep * (1 + tgt_pct) if tgt_pct > 0 else np.inf
            hi_close = ep
            trail = -1.0
            if trail_k > 0 and not np.isnan(atr[i, col]):
                trail = ep - trail_k * atr[i, col]
            xi = -1
            xp = np.nan
            reason = 0
            last_valid = e
            j = e
            while j < n:
                if np.isnan(c[j, col]):
                    j += 1
                    continue
                last_valid = j
                lvl = max(stop, trail)
                if lvl > 0 and l[j, col] <= lvl:
                    xi = j
                    xp = min(o[j, col], lvl) if j > e else lvl
                    reason = 2 if lvl == stop else 4
                    break
                if h[j, col] >= tgt:
                    xi = j
                    xp = max(o[j, col], tgt) if j > e else tgt
                    reason = 3
                    break
                if trail_k > 0:
                    if c[j, col] > hi_close:
                        hi_close = c[j, col]
                    if not np.isnan(atr[j, col]):
                        nt = hi_close - trail_k * atr[j, col]
                        if nt > trail:
                            trail = nt
                held = j - e + 1
                sig = exit_sig[j, col]
                if sig or held >= max_hold:
                    reason = 1 if sig else 0
                    if j + 1 < n and not np.isnan(o[j + 1, col]):
                        xi = j + 1
                        xp = o[j + 1, col]
                    else:
                        xi = j
                        xp = c[j, col]
                        reason = 5
                    break
                j += 1
            if xi < 0:  # ran out of data while holding
                xi = last_valid
                xp = c[last_valid, col]
                reason = 5
            if k >= cap:
                cap *= 2
                out_col = _grow_i(out_col, cap)
                out_ei = _grow_i(out_ei, cap)
                out_xi = _grow_i(out_xi, cap)
                out_ep = _grow_f(out_ep, cap)
                out_xp = _grow_f(out_xp, cap)
                out_reason = _grow_b(out_reason, cap)
            out_col[k] = col
            out_ei[k] = e
            out_xi[k] = xi
            out_ep[k] = ep * (1 + cost)
            out_xp[k] = xp * (1 - cost)
            out_reason[k] = reason
            k += 1
            # next entry signal can be at earliest on the exit day's close
            i = xi if xi > i else i + 1
    return (out_col[:k], out_ei[:k], out_xi[:k], out_ep[:k], out_xp[:k],
            out_reason[:k])


@nb.njit(cache=True)
def _grow_i(a, cap):
    b = np.empty(cap, a.dtype)
    b[: a.size] = a
    return b


@nb.njit(cache=True)
def _grow_f(a, cap):
    b = np.empty(cap, a.dtype)
    b[: a.size] = a
    return b


@nb.njit(cache=True)
def _grow_b(a, cap):
    b = np.empty(cap, a.dtype)
    b[: a.size] = a
    return b


REASONS = np.array(["time", "signal", "stop", "target", "trail", "end"])


def run_trades(panel: Panel, entry: pd.DataFrame, rule: ExitRule,
               exit_sig: pd.DataFrame | None = None, atr: pd.DataFrame | None = None,
               score: pd.DataFrame | None = None, cost: float = 0.0005) -> pd.DataFrame:
    """Simulate every signal on every ticker. Returns one row per trade."""
    cols = panel.tickers
    arr = lambda df: np.ascontiguousarray(df[cols].to_numpy(np.float64))
    o, h, l, c = arr(panel.open), arr(panel.high), arr(panel.low), arr(panel.close)
    ent = np.ascontiguousarray(entry[cols].fillna(False).to_numpy(np.bool_))
    if exit_sig is None:
        ex = np.zeros_like(ent)
    else:
        ex = np.ascontiguousarray(exit_sig[cols].fillna(False).to_numpy(np.bool_))
    at = arr(atr) if atr is not None else np.full_like(c, np.nan)
    col, ei, xi, ep, xp, rs = _simulate(
        o, h, l, c, ent, ex, at, rule.max_hold, rule.stop_pct, rule.target_pct,
        rule.trail_atr, cost,
    )
    dates = panel.dates
    trades = pd.DataFrame({
        "ticker": np.asarray(cols)[col],
        "entry_idx": ei,
        "exit_idx": xi,
        "entry_date": dates[ei],
        "exit_date": dates[xi],
        "entry_px": ep,
        "exit_px": xp,
        "ret": xp / ep - 1,
        "hold": xi - ei,
        "reason": REASONS[rs],
    })
    if score is not None and len(trades):
        s = score[cols].to_numpy(np.float64)
        trades["score"] = s[ei - 1, col]
    return trades.sort_values(["entry_idx", "ticker"], ignore_index=True)
