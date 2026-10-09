"""v2 strategies. Rules are fixed in docs/V2_PLAN.md before any result is looked at."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as ind
from .engine import ExitRule
from .etf import SECTORS
from .signals import Ctx, Strategy


class V2Ctx(Ctx):
    """Ctx plus cross-sectional helpers that need the day's tradable set `uni`."""

    def __init__(self, panel, uni: pd.DataFrame | None = None):
        super().__init__(panel)
        self.uni = uni if uni is not None else pd.DataFrame(True, index=panel.dates, columns=panel.tickers)

    @cached_property
    def vol60(self):
        return self.ret1.rolling(60, min_periods=60).std() * np.sqrt(252)

    @cached_property
    def calm(self):                       # 60d volatility <= median of the day's universe
        med = self.vol60.where(self.uni).median(axis=1)
        return self.vol60.le(med, axis=0)

    @cached_property
    def r126(self):
        return self.c / self.c.shift(126) - 1

    @cached_property
    def leaders(self):                    # top 40% by 126d return within the universe
        q = self.r126.where(self.uni).quantile(0.6, axis=1)
        return self.r126.ge(q, axis=0)

    @cached_property
    def uptrend2(self):
        return (self.c > self.sma(200)) & (self.sma(50) > self.sma(200))


@dataclass
class Rotation:
    name: str
    category: str
    description: str
    score: Callable[[V2Ctx], pd.DataFrame]
    top_n: int
    market_filter: bool = False
    min_score: float | None = None
    only: list[str] | None = None        # restrict to these tickers (e.g. sector ETFs)


@dataclass
class Timing:
    name: str
    category: str
    description: str
    signal: Callable[[V2Ctx], pd.DataFrame]
    only: list[str] | None = None


A, B, C = "稳健回调", "月度轮动", "ETF"


def stock_events() -> list[Strategy]:
    S = []
    for th in (10, 5):
        S.append(Strategy(
            f"强势股回调 RSI2<{th} | 稳健股", A,
            f"成交额前500、60日波动率不高于中位数、50日线>200日线、126日涨幅前40%的股票，RSI(2)<{th}买入；收盘>5日均线卖出，最多8天。",
            lambda x, th=th: x.calm & x.uptrend2 & x.leaders & (x.rsi(2) < th),
            ExitRule(max_hold=8), exit=lambda x: x.c > x.sma(5), score=lambda x: x.rsi(2),
            params={"rsi": th}))
    for n in (3, 4):
        S.append(Strategy(
            f"稳健股连跌{n}天 | 首个上涨日出场", A,
            f"成交额前500、波动率不高于中位数、收盘>200日线的股票连续{n}天收跌买入；首个收涨日卖出，最多8天。",
            lambda x, n=n: x.calm & (x.c > x.sma(200)) & (x.streak >= n),
            ExitRule(max_hold=8), exit=lambda x: x.ret1 > 0, score=lambda x: -x.streak,
            params={"days": n}))
    S.append(Strategy(
        "稳健股急跌6% | 收盘>SMA5出场", A,
        "稳健、上升趋势的股票收盘较前5日最高收盘跌≥6%买入；收盘>5日均线卖出，最多8天。",
        lambda x: x.calm & x.uptrend2 & (x.c <= x.hc(5) * 0.94),
        ExitRule(max_hold=8), exit=lambda x: x.c > x.sma(5), score=lambda x: x.c / x.hc(5)))
    return S


def stock_rotations() -> list[Rotation]:
    R = []
    for n in (20, 30, 50):
        R.append(Rotation(
            f"距52周高点最近 Top{n} + SPY>200日线", B,
            f"每月末在成交额前500里选收盘价最接近252日最高价的{n}只，次日开盘等权买入，持有一个月；SPY低于200日均线时持现金。",
            lambda x: x.c / x.c.rolling(252, min_periods=252).max(), n, market_filter=True))
    for n in (30, 50):
        R.append(Rotation(
            f"12-1月动量 Top{n} + SPY>200日线", B,
            f"每月末在成交额前500里选12个月涨幅（剔除最近21天）最高的{n}只，等权持有一个月；SPY低于200日均线时持现金。",
            lambda x: x.c.shift(21) / x.c.shift(252) - 1, n, market_filter=True))
    for n in (30, 50):
        R.append(Rotation(
            f"低波动 Top{n}", B,
            f"每月末在成交额前500里选60日波动率最低的{n}只，等权持有一个月，不加大盘开关。",
            lambda x: -x.vol60, n))
    return R


def etf_strategies():
    ev = [
        Strategy("SPY RSI2<10 超卖 | 收盘>SMA5出场", C,
                 "SPY收盘>200日均线且RSI(2)<10买入；收盘>5日均线卖出，最多8天。",
                 lambda x: ((x.c > x.sma(200)) & (x.rsi(2) < 10)) & (x.c.columns == "SPY"),
                 ExitRule(max_hold=8), exit=lambda x: x.c > x.sma(5)),
        Strategy("ETF RSI2<10 超卖 | 收盘>SMA5出场", C,
                 "24只ETF各自收盘>200日均线且RSI(2)<10买入；收盘>5日均线卖出，最多8天；最多同时5只。",
                 lambda x: (x.c > x.sma(200)) & (x.rsi(2) < 10), ExitRule(max_hold=8),
                 exit=lambda x: x.c > x.sma(5), score=lambda x: x.rsi(2)),
    ]
    rot = [
        Rotation("ETF 双动量 Top3", C,
                 "每月末按12-1月动量排名取前3，且动量必须大于0，等权持有一个月；不满3只则留现金。",
                 lambda x: x.c.shift(21) / x.c.shift(252) - 1, 3, min_score=0.0),
        Rotation("行业ETF轮动 Top3 + SPY>200日线", C,
                 "每月末在9只行业ETF里选126日涨幅最高的3只，等权持有一个月；SPY低于200日均线时持现金。",
                 lambda x: x.c / x.c.shift(126) - 1, 3, market_filter=True, only=SECTORS),
    ]
    tim = [
        Timing("SPY 200日线择时", C, "SPY收盘>200日均线持有，否则现金。",
               lambda x: x.c > x.sma(200), only=["SPY"]),
        Timing("ETF 趋势组合 (每只>200日线)", C,
               "24只ETF各自收盘>200日均线才持有，等权1/N，其余现金。",
               lambda x: x.c > x.sma(200)),
    ]
    return ev, rot, tim
