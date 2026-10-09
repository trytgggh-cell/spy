"""Run the v3 index / leveraged-ETF strategies (rules and selection: docs/V3_PLAN.md).

    python -m backtest.run_v3
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .leverage import (daily_rf, load_underlying, run_reversion, run_trend, signals,
                       synth_leverage)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3"
PERIODS = {
    "训练 1999–2009": ("1999-03-01", "2009-12-31"),
    "验证 2010–2017": ("2010-01-01", "2017-12-31"),
    "测试 2018 至今": ("2018-01-01", "2100-01-01"),
    "训练+验证 1999–2017": ("1999-03-01", "2017-12-31"),
}


def window_stats(eq: pd.Series, a: str, b: str) -> dict:
    s = eq[(eq.index >= a) & (eq.index <= b)]
    if len(s) < 30:
        return {}
    s = s / s.iloc[0]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    dr = s.pct_change().dropna()
    return {"cagr": s.iloc[-1] ** (1 / yrs) - 1, "mdd": (s / s.cummax() - 1).min(),
            "sharpe": dr.mean() / dr.std() * math.sqrt(252) if dr.std() > 0 else np.nan}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics.OOS_START = pd.Timestamp("2018-01-01")
    u = load_underlying()
    px = u["QQQ"].dropna()
    idx = px.index
    r = px.pct_change().fillna(0.0)
    rf = daily_rf(u["^IRX"]).reindex(idx).ffill().fillna(0.0)
    sg = signals(px)
    rows, yearly, curves, per_rows = [], [], {}, []

    def add(name, cat, desc, eq, trades):
        st = metrics.full_stats(trades, eq) if len(trades) else {**metrics.equity_stats(eq),
                                                                  **metrics.equity_stats(eq[eq.index >= metrics.OOS_START], "pf_oos_")}
        rows.append({"strategy": name, "category": cat, "description": desc, **st})
        if len(trades):
            ys = metrics.yearly_stats(trades); ys["strategy"] = name; yearly.append(ys)
        curves[name] = eq
        for pn, (a, b) in PERIODS.items():
            w = window_stats(eq, a, b)
            if not w:
                continue
            tt = trades[(trades["entry_date"] >= a) & (trades["entry_date"] <= b)] if len(trades) else trades
            per_rows.append({"strategy": name, "period": pn, **w,
                             "win": float((tt["ret"] > 0).mean()) if len(tt) else np.nan, "trades": len(tt)})

    for L in (1, 2, 3):
        cost = 0.0005 if L == 1 else 0.001
        rl = synth_leverage(r, rf, L)
        eq_bh = (1 + rl).cumprod()
        add(f"QQQ {L}倍 买入持有", "基准", f"持有{L}倍QQQ（日杠杆，已扣融资和费用）不动。", eq_bh, pd.DataFrame())
        for key, label in (("T-100", "站上100日线"), ("T-150", "站上150日线"), ("T-200", "站上200日线"),
                           ("T-X", "50日线>200日线")):
            eq, tr = run_trend(sg[key], rl, rf, cost)
            add(f"QQQ {L}倍 | {label}持有", "趋势",
                f"QQQ {'收盘' if key != 'T-X' else ''}{label}时持有{L}倍QQQ，否则持有现金（国债收益）。信号晚一天成交。", eq, tr)
        for th in (10, 5):
            ent = sg["_up200"] & (sg["_rsi2"] < th)
            eq, tr = run_reversion(ent, sg["_above5"], rl, rf, cost, 10)
            add(f"QQQ {L}倍 | RSI2<{th} 超卖反弹", "超卖",
                f"QQQ收盘>200日线且RSI(2)<{th}时买入{L}倍QQQ，收盘>5日线卖出，最多10天；其余时间持有现金。", eq, tr)
    sp = u["SPY"].dropna().reindex(idx).ffill()
    add("SPY 买入持有", "基准", "SPY买入持有（复权）。", sp / sp.iloc[0], pd.DataFrame())

    board = pd.DataFrame(rows)
    board["baseline_wr"] = np.nan
    board["edge_vs_random"] = np.nan
    board["flags"] = ""
    board = board.sort_values("pf_sharpe", ascending=False)
    board.to_csv(OUT / "leaderboard.csv", index=False, float_format="%.6g")
    pd.concat(yearly).to_csv(OUT / "yearly.csv", index=False, float_format="%.6g")
    pd.DataFrame(curves).resample("W-FRI").last().to_csv(OUT / "equity_weekly.csv", float_format="%.6g")
    per = pd.DataFrame(per_rows)
    per.to_csv(OUT / "periods.csv", index=False, float_format="%.6g")
    meta = {"universe": "QQQ 及其 2 倍、3 倍合成杠杆（日杠杆，扣融资和费用）", "n_tickers": 1,
            "start": str(idx[0].date()), "end": str(idx[-1].date()), "oos_start": "2018-01-01",
            "cost_per_side": 0.0005, "max_positions": 1, "top_n": None, "baseline_win_rate": 0.5,
            "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    # ---- pre-registered selection (training + validation only) ----
    tv = per[per["period"] == "训练+验证 1999–2017"].set_index("strategy")
    test = per[per["period"] == "测试 2018 至今"].set_index("strategy")
    trend = [n for n in tv.index if "持有" in n and "买入持有" not in n]
    rev = [n for n in tv.index if "超卖" in n]
    ok_t = tv.loc[trend]; ok_t = ok_t[ok_t["mdd"] >= -0.65]
    pick_t = ok_t["sharpe"].idxmax()
    ok_r = tv.loc[rev]; ok_r = ok_r[ok_r["cagr"] > 0]
    pick_r = ok_r["win"].idxmax()
    print("高收益候选:", pick_t); print(tv.loc[pick_t].drop("period").round(3).to_dict()); print("  测试期:", test.loc[pick_t].drop("period").round(3).to_dict())
    print("高胜率候选:", pick_r); print(tv.loc[pick_r].drop("period").round(3).to_dict()); print("  测试期:", test.loc[pick_r].drop("period").round(3).to_dict())
    (OUT / "picks.json").write_text(json.dumps({"high_return": pick_t, "high_win": pick_r}, ensure_ascii=False))


if __name__ == "__main__":
    main()
