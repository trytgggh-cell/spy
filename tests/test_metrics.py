import numpy as np
import pandas as pd
import pytest

from backtest import metrics


def test_trade_stats_big_wins_and_concentration():
    ret = [0.9, -0.1, -0.1, -0.1, 0.05] + [0.0] * 15          # 20 trades, one +90%
    t = pd.DataFrame({"ret": ret, "hold": 5})
    st = metrics.trade_stats(t)
    assert st["trades"] == 20
    assert st["big_win_rate"] == pytest.approx(1 / 20)
    assert st["best"] == pytest.approx(0.9)
    assert st["top5_share"] == pytest.approx(0.9 / 0.65)       # top 5% = best 1 trade


def test_equity_stats_calendar_years():
    idx = pd.to_datetime(["2020-01-02", "2020-12-31", "2021-12-31", "2022-12-30"])
    eq = pd.Series([1.0, 1.5, 1.2, 1.8], index=idx)
    st = metrics.equity_stats(eq)
    assert st["pf_final"] == pytest.approx(1.8)
    assert st["pf_n_years"] == 3
    assert st["pf_losing_years"] == 1
    assert st["pf_worst_year"] == pytest.approx(1.2 / 1.5 - 1)
    assert st["pf_worst_year_n"] == 2021
    assert st["pf_max_dd"] == pytest.approx(1.2 / 1.5 - 1)
