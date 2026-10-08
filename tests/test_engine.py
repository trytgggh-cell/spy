import numpy as np
import pandas as pd
import pytest

from backtest import indicators as ind
from backtest.data import Panel, synthetic_panel
from backtest.engine import ExitRule, run_trades
from backtest.portfolio import simulate_slots


def make_panel(close, open_=None, high=None, low=None):
    close = np.asarray(close, float)
    n = len(close)
    idx = pd.bdate_range("2020-01-01", periods=n)
    o = np.asarray(open_ if open_ is not None else close, float)
    h = np.asarray(high if high is not None else np.maximum(o, close), float)
    l = np.asarray(low if low is not None else np.minimum(o, close), float)
    df = lambda a: pd.DataFrame({"A": a}, index=idx)
    return Panel(df(o), df(h), df(l), df(close), df(np.full(n, 1e6)))


def sig(p, days):
    e = pd.DataFrame(False, index=p.dates, columns=p.tickers)
    e.iloc[list(days), 0] = True
    return e


def test_entry_next_open_fixed_hold():
    close = np.arange(10, 30, dtype=float)
    open_ = close - 0.5
    p = make_panel(close, open_)
    t = run_trades(p, sig(p, [2]), ExitRule(max_hold=3), cost=0.0)
    assert len(t) == 1
    r = t.iloc[0]
    assert r.entry_idx == 3 and r.entry_px == open_[3]
    # held days 3,4,5 -> sell at open of day 6
    assert r.exit_idx == 6 and r.exit_px == open_[6]
    assert r.reason == "time"


def test_cost_applied_both_sides():
    p = make_panel(np.full(10, 100.0))
    t = run_trades(p, sig(p, [0]), ExitRule(max_hold=2), cost=0.001)
    assert t.iloc[0].ret == pytest.approx(0.999 / 1.001 - 1)


def test_stop_loss_fills_at_stop_or_gap_open():
    close = np.array([100, 100, 100, 99, 90, 85, 85, 85], float)
    open_ = np.array([100, 100, 100, 99.5, 98, 80, 85, 85], float)
    low = np.minimum(open_, close) - 0.1
    p = make_panel(close, open_, low=low)
    t = run_trades(p, sig(p, [1]), ExitRule(max_hold=20, stop_pct=0.05), cost=0.0)
    r = t.iloc[0]
    assert r.entry_px == 100 and r.reason == "stop"
    assert r.exit_idx == 4 and r.exit_px == pytest.approx(95.0)
    # gap through the stop -> filled at the (worse) open
    open2 = open_.copy(); open2[4] = 94; low2 = np.minimum(open2, close) - 0.1
    p2 = make_panel(close, open2, low=low2)
    r2 = run_trades(p2, sig(p2, [1]), ExitRule(max_hold=20, stop_pct=0.05), cost=0.0).iloc[0]
    assert r2.exit_px == 94


def test_exit_signal_and_no_overlap():
    p = make_panel(np.linspace(10, 20, 15))
    ex = sig(p, [5])
    t = run_trades(p, sig(p, [1, 2, 3, 8]), ExitRule(max_hold=50), exit_sig=ex, cost=0.0)
    # signals on 2,3 ignored while holding; exit at open[6]; re-entry from day 8 signal
    assert list(t.entry_idx) == [2, 9]
    assert t.iloc[0].exit_idx == 6 and t.iloc[0].reason == "signal"


def test_no_lookahead_future_change_does_not_alter_past_trades():
    p = synthetic_panel(5, 400, seed=3)
    e = ind.rsi(p.close, 2) < 10
    rule = ExitRule(max_hold=5)
    t1 = run_trades(p, e, rule)
    cut = 300
    p2 = Panel(*(getattr(p, f).copy() for f in ["open", "high", "low", "close", "volume"]))
    for f in ["open", "high", "low", "close"]:
        getattr(p2, f).iloc[cut:] *= 1.5
    e2 = ind.rsi(p2.close, 2) < 10
    t2 = run_trades(p2, e2, rule)
    done = lambda t: t[t.exit_idx < cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(done(t1), done(t2))


def test_rsi_and_streak():
    c = pd.DataFrame({"A": [1, 2, 3, 2, 1, 0.5, 1.0]}, dtype=float)
    assert list(ind.down_streak(c)["A"]) == [0, 0, 0, 1, 2, 3, 0]
    up = pd.DataFrame({"A": np.arange(1, 30, dtype=float)})
    assert ind.rsi(up, 2)["A"].iloc[-1] == 100
    down = pd.DataFrame({"A": np.arange(30, 1, -1, dtype=float)})
    assert ind.rsi(down, 2)["A"].iloc[-1] == pytest.approx(0)


def test_slots_equity_matches_single_trade():
    close = np.array([100, 100, 110, 120, 120, 120], float)
    p = make_panel(close)
    t = run_trades(p, sig(p, [0]), ExitRule(max_hold=3), cost=0.0)
    eq, acc = simulate_slots(p, t, max_pos=1)
    assert len(acc) == 1
    # entered at open[1]=100, exits at open[4]=120
    assert eq.iloc[-1] == pytest.approx(1.2)
    assert eq.iloc[2] == pytest.approx(1.1)
