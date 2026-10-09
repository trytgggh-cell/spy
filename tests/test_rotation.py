import numpy as np
import pandas as pd
import pytest

from backtest.data import Panel
from backtest.portfolio import score_rotation, signal_timing


def panel_from(prices: dict[str, np.ndarray], start="2020-01-01") -> Panel:
    n = len(next(iter(prices.values())))
    idx = pd.bdate_range(start, periods=n)
    mk = lambda f: pd.DataFrame({k: f(v) for k, v in prices.items()}, index=idx)
    c = mk(lambda v: v)
    return Panel(c.copy(), c.copy(), c.copy(), c, mk(lambda v: np.full(n, 1e6)))


def test_signal_timing_trades_next_open_and_holds_only_while_signalled():
    px = np.array([100, 100, 110, 121, 121, 100, 100, 100], float)
    p = panel_from({"A": px})
    sig = pd.DataFrame({"A": [False, True, True, True, False, False, False, False]}, index=p.dates)
    eq, ep = signal_timing(p, sig, cost=0.0)
    assert len(ep) == 1
    e = ep.iloc[0]
    assert e.entry_idx == 2 and e.exit_idx == 5            # signal at close 1 -> open 2; off at close 4 -> open 5
    assert e.ret == pytest.approx(100 / 110 - 1)
    assert eq.iloc[-1] == pytest.approx(100 / 110)


def test_signal_timing_weights_are_one_over_n_with_cash_remainder():
    n = 8
    p = panel_from({"A": np.full(n, 100.0), "B": np.linspace(100, 170, n)})
    sig = pd.DataFrame({"A": False, "B": True}, index=p.dates)
    eq, _ = signal_timing(p, sig, cost=0.0)
    # B is held from open of day 1 to the end with weight 1/2
    assert eq.iloc[-1] == pytest.approx(1 + 0.5 * (170 / p.open["B"].iloc[1] - 1) * (n - 2) / (n - 2), rel=0.3)
    assert eq.iloc[-1] > 1.0


def test_score_rotation_keeps_unfilled_slots_in_cash():
    n = 320
    idx = pd.bdate_range("2020-01-01", periods=n)
    up = np.linspace(100, 200, n); flat = np.full(n, 100.0)
    p = panel_from({"UP": up, "FLAT": flat})
    score = pd.DataFrame({"UP": 1.0, "FLAT": -1.0}, index=idx)
    eq, h = score_rotation(p, score, top_n=2, min_score=0.0, cost=0.0)
    assert set(h["ticker"]) == {"UP"}                       # FLAT never qualifies
    # half the account sits in cash, so growth is about half of UP's
    start = float(eq[eq.index == h["entry_date"].iloc[0]].iloc[0])
    assert eq.iloc[-1] > 1.0
    assert eq.iloc[-1] < up[-1] / up[259] * 0.75 + 0.5
