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
        "edge_vs_random"]


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


def build() -> Path:
    board = pd.read_csv(RES / "leaderboard.csv")
    meta = json.loads((RES / "meta.json").read_text())
    yearly = pd.read_csv(RES / "yearly.csv")
    eq = pd.read_csv(RES / "equity_weekly.csv", index_col=0, parse_dates=True)

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
    data = {
        "meta": meta,
        "rows": rows,
        "yearly": yd,
        "dates": [d.strftime("%Y-%m-%d") for d in eq.index],
        "curves": curves,
        "recommended": pick_recommendations(board),
        "findings": (json.loads((RES / "findings.json").read_text(encoding="utf-8"))
                     if (RES / "findings.json").exists() else []),
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(build())
