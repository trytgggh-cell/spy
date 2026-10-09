"""Run the v2 strategies (rules in docs/V2_PLAN.md).

    python -m backtest.run_v2 --universe market   # stocks: events + monthly rotations
    python -m backtest.run_v2 --universe etf      # ETF basket
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import etf, market, metrics
from .data import load_or_download
from .engine import run_trades
from .indicators import sma
from .portfolio import score_rotation, signal_timing, simulate_slots
from .run_all import _pct, flag
from .signals import build_strategies
from .strategies_v2 import V2Ctx, etf_strategies, stock_events, stock_rotations

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["market", "etf"], required=True)
    ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--oos-start", default="2015-01-01")
    args = ap.parse_args()
    metrics.OOS_START = pd.Timestamp(args.oos_start)
    out = ROOT / "results" / f"v2_{args.universe}"
    (out / "trades").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    idx = load_or_download([])
    if args.universe == "market":
        panel, uni = market.build_market_panel(500)
        label = "全市场当前仍上市的美股，每个时点取成交额前500名（不看指数成分）"
        events, rotations, timings, max_pos = stock_events(), stock_rotations(), [], 10
        events += [s for s in build_strategies() if s.name.startswith("随机入场")]   # baselines
    else:
        panel, uni = etf.load_etf_panel(), None
        label = "24只ETF（股票、行业、债券、商品）"
        events, rotations, timings = [], [], []
        ev, rot, tim = etf_strategies()
        events, rotations, timings, max_pos = ev, rot, tim, 5
    dates = panel.dates
    ctx = V2Ctx(panel, uni)
    spy_full = idx.close["SPY"].reindex(dates).ffill()
    spy_on = (spy_full > sma(spy_full.to_frame(), 200).iloc[:, 0]).fillna(False)
    meta = {"universe": label, "n_tickers": len(panel.tickers), "start": str(dates[0].date()),
            "end": str(dates[-1].date()), "oos_start": args.oos_start, "cost_per_side": args.cost,
            "max_positions": max_pos, "top_n": 500 if uni is not None else None}
    print(f"panel {panel.close.shape} {meta['start']}..{meta['end']} in {time.time() - t0:.0f}s", flush=True)

    rows, yearly, curves = [], [], {}

    def add(name, cat, desc, trades, eq):
        st = metrics.full_stats(trades, eq)
        rows.append({"strategy": name, "category": cat, "description": desc, **st})
        ys = metrics.yearly_stats(trades); ys["strategy"] = name; yearly.append(ys)
        curves[name] = eq
        print(f"  {name:<40} n={st.get('trades', 0):>6} win={_pct(st.get('win_rate'))} "
              f"cagr={_pct(st.get('pf_cagr'))} dd={_pct(st.get('pf_max_dd'), '.0%')}", flush=True)

    for k, s in enumerate(events, 1):
        ent = s.entry(ctx)
        if uni is not None:
            ent = ent & uni
        ex = s.exit(ctx) if s.exit else None
        sc = s.score(ctx) if s.score else None
        tr = run_trades(panel, ent, s.rule, ex, ctx.atr14 if s.use_atr else None, sc, args.cost)
        eq, _ = simulate_slots(panel, tr, max_pos)
        add(s.name, s.category, s.description, tr, eq)
    for r in rotations:
        p = panel.subset([t for t in panel.tickers if t in r.only]) if r.only else panel
        c2 = V2Ctx(p, None) if r.only else ctx
        eq, h = score_rotation(p, r.score(c2), r.top_n, spy_on if r.market_filter else None,
                               uni if (uni is not None and not r.only) else None, args.cost,
                               min_score=r.min_score)
        add(r.name, r.category, r.description, h, eq)
    for t in timings:
        p = panel.subset(t.only) if t.only else panel
        c2 = V2Ctx(p, None)
        eq, ep = signal_timing(p, t.signal(c2), args.cost)
        add(t.name, t.category, t.description, ep, eq)

    for b in ("SPY", "QQQ"):
        px = idx.close[b].reindex(dates).ffill()
        eq = px / px.dropna().iloc[0]
        st = {**metrics.equity_stats(eq), **metrics.equity_stats(eq[eq.index >= metrics.OOS_START], "pf_oos_")}
        rows.append({"strategy": f"{b} 买入持有", "category": "基准", "description": f"{b} 买入持有（复权）。", **st})
        curves[f"{b} 买入持有"] = eq

    board = pd.DataFrame(rows)
    rnd = board[board["strategy"].str.match(r"随机入场 \| 持有\d+天")]
    if len(rnd):
        base_by_hold = dict(zip(rnd["avg_hold"], rnd["win_rate"]))
        meta["baseline_win_rate"] = float(rnd.loc[rnd["avg_hold"].idxmin(), "win_rate"])
        board["baseline_wr"] = board["avg_hold"].map(
            lambda h: np.nan if not np.isfinite(h) else base_by_hold[min(base_by_hold, key=lambda k: abs(np.log(k) - np.log(max(h, 1))))])
        board.loc[board["category"] == "基线", "baseline_wr"] = np.nan
    else:
        meta["baseline_win_rate"] = 0.5
        board["baseline_wr"] = np.nan
    board["flags"] = board.apply(flag, axis=1)
    board["edge_vs_random"] = board["win_rate"] - board["baseline_wr"]
    board = board.sort_values("pf_sharpe", ascending=False, na_position="last")
    board.to_csv(out / "leaderboard.csv", index=False, float_format="%.6g")
    pd.concat(yearly).to_csv(out / "yearly.csv", index=False, float_format="%.6g")
    pd.DataFrame(curves).resample("W-FRI").last().to_csv(out / "equity_weekly.csv", float_format="%.6g")
    meta["generated"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
