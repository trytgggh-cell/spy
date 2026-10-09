"""Build report/index.html from results/ (single file, data embedded).

    python -m backtest.report
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "report" / "index.html"
TEMPLATE = Path(__file__).with_name("report_template.html")

COLS = ["strategy", "category", "description", "trades", "win_rate", "avg_ret",
        "median_ret", "avg_win", "avg_loss", "payoff", "profit_factor", "worst",
        "t_stat", "avg_hold", "is_trades", "is_win_rate", "is_avg_ret",
        "oos_trades", "oos_win_rate", "oos_avg_ret", "oos_profit_factor",
        "years_positive", "yearly_win_rate_min", "pf_cagr", "pf_sharpe",
        "pf_max_dd", "pf_oos_cagr", "pf_oos_max_dd", "flags", "baseline_wr",
        "edge_vs_random", "big_win_rate", "best", "top5_share",
        "pf_final", "pf_worst_year", "pf_worst_year_n", "pf_losing_years", "pf_n_years",
        "pf_oos_final", "pf_oos_worst_year", "pf_oos_worst_year_n",
        "pf_oos_losing_years", "pf_oos_n_years"]

# Strategies shown as "what happens to $10,000" cards: (group, name, note)
BIAS = "股票池只含今天的成分股，这个数字明显偏高，不是真实可得的收益。"
CURATED = [
    ("对照", "SPY 买入持有", ""),
    ("对照", "QQQ 买入持有", ""),
    ("高胜率", "5日内从高点回撤>8% 上升趋势 | 收盘>SMA5出场", ""),
    ("高胜率", "RSI2<5 上升趋势 | 收盘>SMA5出场 | 最长10天", ""),
    ("高胜率", "跌破布林下轨 上升趋势 | 回到中轨出场", ""),
    ("大赚型", "50/200均线金叉 | 死叉出场", "依赖少数大赢家，受今天成分股的偏差影响。"),
    ("大赚型", "55日新高 | 6ATR追踪止损", ""),
    ("大赚型", "52周新高 | 8ATR追踪止损", "依赖少数大赢家，受今天成分股的偏差影响。"),
    ("大赚型", "20日新高突破(海龟) | 3ATR追踪止损", "止损太紧，大赚的机会被提前止损掉了。"),
    ("动量", "动量轮动 12个月动量 Top10", BIAS),
]


def pick_recommendations(board: pd.DataFrame, k: int = 5) -> list[str]:
    """Stable, high-win-rate, positive-expectancy strategies (IS and OOS)."""
    b = board[~board["category"].isin(["基线", "基准"])].copy()
    ok = (
        (b["trades"] >= 200) & (b["oos_trades"] >= 50)
        & (b["avg_ret"] > 0) & (b["oos_avg_ret"] > 0)
        & (b["profit_factor"] > 1.2)
        & (b["win_rate"] >= b["baseline_wr"] + 0.05)
        & (b["oos_win_rate"] >= b["baseline_wr"] + 0.03)
    )
    b = b[ok].sort_values(["win_rate", "profit_factor"], ascending=False)
    return b["strategy"].head(k).tolist()


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(v) else round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


# Card notes that depend on how the stock pool was built
MARKET_NOTES = {
    "50/200均线金叉 | 死叉出场": "依赖少数大赢家；股票池不含已退市公司，收益仍偏高。",
    "52周新高 | 8ATR追踪止损": "依赖少数大赢家；股票池不含已退市公司，收益仍偏高。",
    "动量轮动 12个月动量 Top10": "股票池不含已退市公司，这个数字仍是上限。",
}


def build_data(res: Path, market: bool = False) -> dict | None:
    if not (res / "leaderboard.csv").exists():
        return None
    board = pd.read_csv(res / "leaderboard.csv")
    meta = json.loads((res / "meta.json").read_text())
    yearly = pd.read_csv(res / "yearly.csv")
    eq = pd.read_csv(res / "equity_weekly.csv", index_col=0, parse_dates=True)

    for c in COLS:
        if c not in board:
            board[c] = np.nan
    board["flags"] = board["flags"].fillna("")
    rows = [{k: _clean(v) for k, v in r.items()} for r in board[COLS].to_dict("records")]

    yd: dict[str, list] = {}
    for name, g in yearly.groupby("strategy"):
        yd[name] = [[int(r.year), int(r.trades), _clean(r.win_rate), _clean(r.avg_ret)]
                    for r in g.itertuples()]

    eq = eq.ffill()
    curves = {c: [_clean(round(v, 4)) for v in eq[c].to_numpy()] for c in eq.columns}
    extra = {}
    if (res / "notes.json").exists():
        extra = json.loads((res / "notes.json").read_text(encoding="utf-8"))
    return {
        "meta": meta,
        "rows": rows,
        "yearly": yd,
        "dates": [d.strftime("%Y-%m-%d") for d in eq.index],
        "curves": curves,
        "recommended": pick_recommendations(board),
        "curated": [{"group": g, "name": n, "note": (MARKET_NOTES.get(n, note) if market else note)}
                    for g, n, note in CURATED if n in set(board["strategy"])],
        "findings": (json.loads((res / "findings.json").read_text(encoding="utf-8"))
                     if (res / "findings.json").exists() else []),
        **extra,
    }


def build() -> Path:
    data = {"index": build_data(RES), "market": build_data(RES / "market", market=True)}
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(build())
