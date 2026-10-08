"""Run every strategy and write results/ (leaderboard, yearly stats, equity curves).

    python -m backtest.run_all              # real data (downloads on first run)
    python -m backtest.run_all --synthetic  # offline smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .data import Panel, load_or_download, synthetic_panel
from .engine import run_trades
from .indicators import sma
from .portfolio import momentum_rotation, simulate_slots
from .signals import Ctx, build_strategies
from .universe import BENCHMARKS, get_universe

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def _pct(v, fmt: str = ".1%") -> str:
    return "—" if v is None or not np.isfinite(v) else format(v, fmt)


def flag(row: pd.Series) -> str:
    notes = []
    baseline_wr = row.get("baseline_wr", np.nan)
    if row.get("trades", 0) < 200:
        notes.append("样本少")
    if row.get("oos_trades", 0) >= 30:
        if row.get("oos_avg_ret", 0) <= 0:
            notes.append("样本外亏损")
        elif row.get("oos_win_rate", 0) < row.get("is_win_rate", 0) - 0.05:
            notes.append("样本外胜率下滑")
    if row.get("avg_ret", 0) <= 0:
        notes.append("负期望")
    if np.isfinite(baseline_wr) and row.get("win_rate", 0) < baseline_wr + 0.02:
        notes.append("不优于随机")
    return "、".join(notes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-download data")
    ap.add_argument("--max-pos", type=int, default=10)
    ap.add_argument("--cost", type=float, default=0.0005, help="per side")
    ap.add_argument("--start", default="2004-01-01")
    args = ap.parse_args()

    t0 = time.time()
    if args.synthetic:
        full = synthetic_panel(60, 4000, start="2005-01-03")
        stocks, bench = full, {"SPY": full.close.mean(axis=1)}
        meta = {"universe": "synthetic", "n_tickers": len(full.tickers)}
    else:
        uni = get_universe()
        full = load_or_download(uni["ticker"].tolist() + BENCHMARKS, args.start,
                                args.refresh)
        names = [t for t in full.tickers if t not in BENCHMARKS]
        stocks = full.subset(names)
        bench = {b: full.close[b] for b in BENCHMARKS if b in full.tickers}
        meta = {"universe": "S&P 500 + 纳指100（当前成分股）",
                "n_tickers": len(names)}
    dates = stocks.dates
    meta.update(start=str(dates[0].date()), end=str(dates[-1].date()),
                oos_start=str(metrics.OOS_START.date()), cost_per_side=args.cost,
                max_positions=args.max_pos)
    print(f"panel {stocks.close.shape} {meta['start']}..{meta['end']} "
          f"loaded in {time.time() - t0:.0f}s")

    ctx = Ctx(stocks)
    RES.mkdir(exist_ok=True)
    (RES / "trades").mkdir(exist_ok=True)
    rows, yearly, curves = [], [], {}

    for s in build_strategies():
        t1 = time.time()
        entry = s.entry(ctx)
        ex = s.exit(ctx) if s.exit else None
        sc = s.score(ctx) if s.score else None
        trades = run_trades(stocks, entry, s.rule, ex,
                            ctx.atr14 if s.use_atr else None, sc, args.cost)
        eq, _ = simulate_slots(stocks, trades, args.max_pos)
        st = metrics.full_stats(trades, eq)
        rows.append({"strategy": s.name, "category": s.category,
                     "description": s.description, **st})
        ys = metrics.yearly_stats(trades)
        ys["strategy"] = s.name
        yearly.append(ys)
        curves[s.name] = eq
        trades.to_parquet(RES / "trades" / f"{len(rows):02d}.parquet", index=False)
        print(f"  {s.name:<45} n={st.get('trades', 0):>7} "
              f"win={_pct(st.get('win_rate'))} avg={_pct(st.get('avg_ret'), '+.2%')} "
              f"({time.time() - t1:.1f}s)")

    # ---------------- momentum rotation ----------------------------------------
    spy = bench.get("SPY")
    spy_filter = (spy > sma(spy.to_frame(), 200).iloc[:, 0]).reindex(dates).fillna(False) \
        if spy is not None else None
    for lb, skip, top, filt in ((252, 21, 10, False), (252, 21, 20, False),
                                (252, 21, 10, True), (126, 5, 10, True),
                                (63, 5, 10, True)):
        name = (f"动量轮动 {lb // 21}个月动量 Top{top}"
                + (" + SPY>200日均线过滤" if filt else ""))
        eq, hold = momentum_rotation(stocks, lb, skip, top,
                                     spy_filter if filt else None, args.cost)
        st = metrics.full_stats(hold, eq)
        desc = (f"每月末按过去{lb // 21}个月涨幅(剔除最近{skip}天)选最强{top}只，次日开盘等权买入，"
                f"持有一个月" + ("；SPY低于200日均线时空仓" if filt else "") + "。")
        rows.append({"strategy": name, "category": "动量轮动", "description": desc, **st})
        ys = metrics.yearly_stats(hold)
        ys["strategy"] = name
        yearly.append(ys)
        curves[name] = eq
        print(f"  {name:<45} n={st.get('trades', 0):>7} win={_pct(st.get('win_rate'))} "
              f"cagr={_pct(st.get('pf_cagr'))}")

    # ---------------- benchmarks -------------------------------------------------
    for b, px in bench.items():
        px = px.reindex(dates).ffill()
        first = px.first_valid_index()
        eq = (px / px.loc[first]).fillna(1.0)
        st = {**metrics.equity_stats(eq), **metrics.equity_stats(eq[eq.index >= metrics.OOS_START], "pf_oos_")}
        rows.append({"strategy": f"{b} 买入持有", "category": "基准",
                     "description": f"{b} 买入持有（复权）。", **st})
        curves[f"{b} 买入持有"] = eq

    board = pd.DataFrame(rows)
    # Random entries win more often the longer they are held, so compare each
    # strategy with the random baseline whose holding period is closest.
    rnd = board[board["strategy"].str.match(r"随机入场 \| 持有\d+天")]
    base_by_hold = dict(zip(rnd["avg_hold"], rnd["win_rate"]))
    base_wr = float(rnd.loc[rnd["avg_hold"].idxmin(), "win_rate"]) if len(rnd) else 0.5

    def matched(h):
        if not base_by_hold or not np.isfinite(h):
            return np.nan
        return base_by_hold[min(base_by_hold, key=lambda k: abs(np.log(k) - np.log(max(h, 1))))]

    board["baseline_wr"] = board["avg_hold"].map(matched)
    board.loc[board["category"] == "基线", "baseline_wr"] = np.nan
    board["flags"] = board.apply(flag, axis=1)
    board["edge_vs_random"] = board["win_rate"] - board["baseline_wr"]
    board = board.sort_values("win_rate", ascending=False, na_position="last")
    board.to_csv(RES / "leaderboard.csv", index=False, float_format="%.6g")
    pd.concat(yearly).to_csv(RES / "yearly.csv", index=False, float_format="%.6g")
    weekly = pd.DataFrame(curves).resample("W-FRI").last()
    weekly.to_csv(RES / "equity_weekly.csv", float_format="%.6g")
    meta["baseline_win_rate"] = base_wr
    meta["generated"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    (RES / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {RES}")


if __name__ == "__main__":
    main()
