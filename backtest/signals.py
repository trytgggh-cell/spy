"""Strategy catalogue.

Each Strategy turns the price panel into an entry matrix (bool, evaluated at
the close), an optional exit matrix, an ExitRule and an optional ranking score
(lower = preferred when the portfolio has more signals than free slots).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as ind
from .data import Panel
from .engine import ExitRule


class Ctx:
    """Lazily computed indicator cache shared by all strategies."""

    def __init__(self, p: Panel):
        self.p = p
        self.o, self.h, self.l, self.c, self.v = p.open, p.high, p.low, p.close, p.volume
        self._cache: dict = {}

    def get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def sma(self, n):
        return self.get(("sma", n), lambda: ind.sma(self.c, n))

    def rsi(self, n):
        return self.get(("rsi", n), lambda: ind.rsi(self.c, n))

    def hh(self, n):  # highest high of the previous n days (excl. today)
        return self.get(("hh", n), lambda: ind.highest(self.h, n).shift(1))

    def hc(self, n):  # highest close of previous n days
        return self.get(("hc", n), lambda: ind.highest(self.c, n).shift(1))

    def ll(self, n):
        return self.get(("ll", n), lambda: ind.lowest(self.l, n).shift(1))

    def vol_avg(self, n):
        return self.get(("vavg", n), lambda: ind.sma(self.v, n).shift(1))

    @cached_property
    def atr14(self):
        return ind.atr(self.h, self.l, self.c, 14)

    @cached_property
    def ibs(self):
        return ind.ibs(self.h, self.l, self.c)

    @cached_property
    def streak(self):
        return ind.down_streak(self.c)

    @cached_property
    def ret1(self):
        return self.c.pct_change(fill_method=None)

    @cached_property
    def uptrend(self):
        return self.c > self.sma(200)

    @cached_property
    def liquid(self):
        """Price >= $5 and 20d average dollar volume >= $10M."""
        dv = (self.c * self.v).rolling(20, min_periods=20).mean()
        return (self.c >= 5) & (dv >= 1e7)

    @cached_property
    def boll(self):
        return ind.bollinger(self.c, 20, 2.0)

    @cached_property
    def macd(self):
        return ind.macd(self.c)


@dataclass
class Strategy:
    name: str
    category: str
    description: str
    entry: Callable[[Ctx], pd.DataFrame]
    rule: ExitRule
    exit: Callable[[Ctx], pd.DataFrame] | None = None
    score: Callable[[Ctx], pd.DataFrame] | None = None
    use_atr: bool = False
    params: dict = field(default_factory=dict)


MR, TR, PT = "均值回归", "趋势突破", "形态事件"


def build_strategies() -> list[Strategy]:
    S: list[Strategy] = []

    # ---------------- short-term mean reversion (long only, uptrend filter) ----
    for th in (5, 10):
        for mh in (5, 10):
            S.append(Strategy(
                f"RSI2<{th} 上升趋势 | 收盘>SMA5出场 | 最长{mh}天", MR,
                f"收盘>200日均线且RSI(2)<{th}时次日开盘买入；收盘站上5日均线后次日开盘卖出，最多持有{mh}天。",
                lambda x, th=th: x.uptrend & (x.rsi(2) < th) & x.liquid,
                ExitRule(max_hold=mh),
                exit=lambda x: x.c > x.sma(5),
                score=lambda x: x.rsi(2),
                params={"rsi_th": th, "max_hold": mh},
            ))
    S.append(Strategy(
        "RSI2<10 无趋势过滤 | 收盘>SMA5出场", MR,
        "不加200日均线过滤的RSI(2)<10，用于对比趋势过滤的作用。",
        lambda x: (x.rsi(2) < 10) & x.liquid, ExitRule(max_hold=10),
        exit=lambda x: x.c > x.sma(5), score=lambda x: x.rsi(2),
    ))
    S.append(Strategy(
        "RSI2<10 上升趋势 | 收盘>SMA5出场 | 8%止损", MR,
        "同RSI2<10，但加8%硬止损，观察止损对胜率与期望的影响。",
        lambda x: x.uptrend & (x.rsi(2) < 10) & x.liquid, ExitRule(max_hold=10, stop_pct=0.08),
        exit=lambda x: x.c > x.sma(5), score=lambda x: x.rsi(2),
    ))
    for n in (3, 4, 5):
        S.append(Strategy(
            f"连跌{n}天 上升趋势 | 首个上涨日出场", MR,
            f"收盘>200日均线且连续{n}天收跌，次日开盘买入；出现收涨日后次日开盘卖出，最多10天。",
            lambda x, n=n: x.uptrend & (x.streak >= n) & x.liquid,
            ExitRule(max_hold=10), exit=lambda x: x.ret1 > 0,
            score=lambda x: -x.streak, params={"days": n},
        ))
    for n in (3, 4):
        S.append(Strategy(
            f"连跌{n}天 上升趋势 | 持有5天", MR,
            f"收盘>200日均线且连续{n}天收跌，次日开盘买入，固定持有5个交易日。",
            lambda x, n=n: x.uptrend & (x.streak >= n) & x.liquid,
            ExitRule(max_hold=5), score=lambda x: -x.streak, params={"days": n},
        ))
    S.append(Strategy(
        "跌破布林下轨 上升趋势 | 回到中轨出场", MR,
        "收盘>200日均线且收盘跌破布林带(20,2)下轨，次日买入；收盘回到20日均线上方卖出，最多15天。",
        lambda x: x.uptrend & (x.c < x.boll[0]) & x.liquid,
        ExitRule(max_hold=15), exit=lambda x: x.c > x.boll[1],
        score=lambda x: (x.c - x.boll[0]) / x.c,
    ))
    for th in (0.1, 0.2):
        S.append(Strategy(
            f"IBS<{th} 上升趋势 | IBS>0.7出场", MR,
            f"收盘>200日均线且当日收盘位于日内区间底部(IBS<{th})，次日买入；IBS>0.7的次日卖出，最多5天。",
            lambda x, th=th: x.uptrend & (x.ibs < th) & x.liquid,
            ExitRule(max_hold=5), exit=lambda x: x.ibs > 0.7, score=lambda x: x.ibs,
            params={"ibs": th},
        ))
    S.append(Strategy(
        "IBS<0.2 且 RSI2<10 上升趋势 | 收盘>SMA5出场", MR,
        "两个超卖条件叠加：IBS<0.2且RSI(2)<10，收盘>200日均线。",
        lambda x: x.uptrend & (x.ibs < 0.2) & (x.rsi(2) < 10) & x.liquid,
        ExitRule(max_hold=10), exit=lambda x: x.c > x.sma(5), score=lambda x: x.rsi(2),
    ))
    for dd in (0.08, 0.12):
        S.append(Strategy(
            f"5日内从高点回撤>{int(dd*100)}% 上升趋势 | 收盘>SMA5出场", MR,
            f"收盘>200日均线，收盘较前5日最高收盘下跌超过{int(dd*100)}%，次日买入。",
            lambda x, dd=dd: x.uptrend & (x.c < x.hc(5) * (1 - dd)) & x.liquid,
            ExitRule(max_hold=10), exit=lambda x: x.c > x.sma(5),
            score=lambda x: x.c / x.hc(5), params={"drawdown": dd},
        ))

    # ---------------- trend / breakout ----------------------------------------
    S.append(Strategy(
        "50/200均线金叉 | 死叉出场", TR,
        "50日均线上穿200日均线买入，下穿卖出（经典黄金交叉）。",
        lambda x: ind.cross_above(x.sma(50), x.sma(200)) & x.liquid,
        ExitRule(max_hold=2000), exit=lambda x: x.sma(50) < x.sma(200),
    ))
    for n, k in ((20, 3.0), (55, 3.0)):
        S.append(Strategy(
            f"{n}日新高突破(海龟) | {k:g}ATR追踪止损", TR,
            f"收盘突破前{n}日最高价买入；用最高收盘价-{k:g}倍ATR的吊灯止损离场。",
            lambda x, n=n: (x.c > x.hh(n)) & x.uptrend & x.liquid,
            ExitRule(max_hold=500, trail_atr=k), use_atr=True,
            score=lambda x: -(x.c / x.sma(200)), params={"n": n, "atr": k},
        ))
    S.append(Strategy(
        "20日新高突破 | 跌破10日低点出场", TR,
        "海龟系统一：收盘突破20日最高价买入，跌破10日最低价卖出。",
        lambda x: (x.c > x.hh(20)) & x.uptrend & x.liquid,
        ExitRule(max_hold=500), exit=lambda x: x.c < x.ll(10),
        score=lambda x: -(x.c / x.sma(200)),
    ))
    S.append(Strategy(
        "MACD零轴上方金叉 | MACD死叉出场", TR,
        "MACD线在零轴上方上穿信号线买入，下穿信号线卖出。",
        lambda x: ind.cross_above(*x.macd) & (x.macd[0] > 0) & x.liquid,
        ExitRule(max_hold=250), exit=lambda x: x.macd[0] < x.macd[1],
    ))
    S.append(Strategy(
        "52周新高+放量 | 持有20天", TR,
        "收盘创252日新高且成交量>1.5倍20日均量，次日买入持有20个交易日。",
        lambda x: (x.c > x.hc(252)) & (x.v > 1.5 * x.vol_avg(20)) & x.liquid,
        ExitRule(max_hold=20), score=lambda x: -x.v / x.vol_avg(20),
    ))
    S.append(Strategy(
        "52周新高+放量 | 3ATR追踪止损", TR,
        "收盘创252日新高且放量，次日买入，3倍ATR吊灯止损离场。",
        lambda x: (x.c > x.hc(252)) & (x.v > 1.5 * x.vol_avg(20)) & x.liquid,
        ExitRule(max_hold=500, trail_atr=3.0), use_atr=True,
        score=lambda x: -x.v / x.vol_avg(20),
    ))

    # wide trailing stops: fewer, bigger winners (low win rate, high payoff)
    def wide(name, entry, k, desc):
        S.append(Strategy(
            name, TR, desc, entry, ExitRule(max_hold=3000, trail_atr=k), use_atr=True,
            score=lambda x: -(x.c / x.sma(200)), params={"atr": k},
        ))

    wide("20日新高 | 5ATR追踪止损", lambda x: (x.c > x.hh(20)) & x.uptrend & x.liquid, 5.0,
         "收盘创20日新高且在200日均线上方买入；最高收盘价-5倍ATR的吊灯止损离场（比海龟的3倍更宽）。")
    wide("55日新高 | 6ATR追踪止损", lambda x: (x.c > x.hh(55)) & x.uptrend & x.liquid, 6.0,
         "收盘创55日新高且在200日均线上方买入；最高收盘价-6倍ATR的吊灯止损离场。")
    wide("52周新高 | 5ATR追踪止损", lambda x: (x.c > x.hc(252)) & x.liquid, 5.0,
         "收盘创252日新高买入（不要求放量）；5倍ATR吊灯止损离场。")
    wide("52周新高 | 8ATR追踪止损", lambda x: (x.c > x.hc(252)) & x.liquid, 8.0,
         "收盘创252日新高买入；8倍ATR吊灯止损离场，让赢家尽量跑。")
    wide("52周新高+放量 | 5ATR追踪止损",
         lambda x: (x.c > x.hc(252)) & (x.v > 1.5 * x.vol_avg(20)) & x.liquid, 5.0,
         "收盘创252日新高且成交量>1.5倍20日均量买入；5倍ATR吊灯止损离场。")

    # ---------------- patterns / events -----------------------------------------
    for g in (0.03, 0.05):
        S.append(Strategy(
            f"跳空低开>{int(g*100)}%收阳 上升趋势 | 持有5天", PT,
            f"开盘较前收低开超过{int(g*100)}%，但收盘高于开盘（缺口后承接），次日买入持有5天。",
            lambda x, g=g: x.uptrend & (x.o < x.c.shift(1) * (1 - g)) & (x.c > x.o) & x.liquid,
            ExitRule(max_hold=5), score=lambda x: x.o / x.c.shift(1), params={"gap": g},
        ))
    S.append(Strategy(
        "跳空低开>3% 上升趋势 | 收盘>SMA5出场", PT,
        "收盘>200日均线且当日跳空低开超过3%，次日开盘买入，反弹站上5日均线卖出。",
        lambda x: x.uptrend & (x.o < x.c.shift(1) * 0.97) & x.liquid,
        ExitRule(max_hold=10), exit=lambda x: x.c > x.sma(5),
        score=lambda x: x.o / x.c.shift(1),
    ))
    S.append(Strategy(
        "放量突破20日高点 | 持有10天", PT,
        "收盘突破前20日最高价且成交量>2倍20日均量，次日买入持有10天。",
        lambda x: (x.c > x.hh(20)) & (x.v > 2 * x.vol_avg(20)) & x.liquid,
        ExitRule(max_hold=10), score=lambda x: -x.v / x.vol_avg(20),
    ))
    S.append(Strategy(
        "内包线 上升趋势 | 持有5天", PT,
        "当日最高低于前日最高、最低高于前日最低（内包线），且收盘>200日均线，次日买入持有5天。",
        lambda x: x.uptrend & (x.h < x.h.shift(1)) & (x.l > x.l.shift(1)) & x.liquid,
        ExitRule(max_hold=5),
    ))
    for d in (0.05, 0.08):
        S.append(Strategy(
            f"单日大跌>{int(d*100)}% 上升趋势 | 持有5天", PT,
            f"收盘>200日均线的股票单日下跌超过{int(d*100)}%，次日买入持有5天（博反弹）。",
            lambda x, d=d: x.uptrend & (x.ret1 < -d) & x.liquid,
            ExitRule(max_hold=5), score=lambda x: x.ret1, params={"drop": d},
        ))
    S.append(Strategy(
        "单日大跌>5% 无趋势过滤 | 持有5天", PT,
        "任何股票单日下跌超过5%，次日买入持有5天（对比趋势过滤）。",
        lambda x: (x.ret1 < -0.05) & x.liquid, ExitRule(max_hold=5), score=lambda x: x.ret1,
    ))

    # ---------------- baselines -------------------------------------------------
    for mh in (5, 10, 20):
        S.append(Strategy(
            f"随机入场 | 持有{mh}天 (基线)", "基线",
            f"每天随机抽取约2%的股票买入并持有{mh}天，代表“随便买”的胜率。",
            lambda x: _random_mask(x.c, 0.02) & x.liquid, ExitRule(max_hold=mh),
            params={"max_hold": mh},
        ))
    S.append(Strategy(
        "随机入场 上升趋势 | 持有5天 (基线)", "基线",
        "只在收盘>200日均线的股票里随机买入持有5天，用来衡量趋势过滤本身的贡献。",
        lambda x: _random_mask(x.c, 0.02) & x.uptrend & x.liquid, ExitRule(max_hold=5),
    ))
    return S


def _random_mask(like: pd.DataFrame, p: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random(like.shape) < p, index=like.index, columns=like.columns)
