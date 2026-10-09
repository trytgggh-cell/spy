# 美股打法胜率回测

在 S&P 500 + 纳指100（约 517 只）上回测 ~40 个短线/趋势/形态/动量打法，找出**胜率高且长期期望为正**的规则，并生成交互式报告。

## 运行

```bash
pip install -r requirements.txt
python -m backtest.run_all          # 首次运行会下载 2004 年至今的日线（yfinance），缓存到 data/
python -m backtest.report           # 生成 report/index.html
python -m pytest tests              # 引擎单元测试
```

离线试跑（随机生成的价格，只用于验证流程）：`python -m backtest.run_all --synthetic`

常用参数：`--max-pos 10`（组合最多持仓数）、`--cost 0.0005`（单边成本）、`--refresh`（重新下载数据）。

需要能访问 `query1.finance.yahoo.com` / `query2.finance.yahoo.com`（失败时回退 `stooq.com`）和 `raw.githubusercontent.com`（S&P 500 成分股列表）。

## 策略

| 类别 | 打法 |
|---|---|
| 均值回归 | RSI(2) 超卖、连跌 N 天、跌破布林下轨、低 IBS、5 日内急跌，均加 200 日均线趋势过滤 |
| 趋势突破 | 50/200 金叉、20/55 日新高 + ATR 追踪止损（3 倍的海龟版，以及 5–8 倍的宽止损版）、海龟 20/10、MACD 零轴上金叉、52 周新高放量 |
| 形态事件 | 跳空低开收阳、放量突破、内包线、单日大跌后反弹 |
| 动量轮动 | 12/6/3 个月动量 Top N 月度轮动，可选 SPY 200 日均线择时 |
| 基线 | 随机入场持有 5/10/20 天，用来衡量"随便买"的胜率 |

策略定义在 `backtest/signals.py`，新增一个打法只需在 `build_strategies()` 里加一个 `Strategy`。

## 回测规则

- 收盘出信号，**次日开盘**成交；出场信号同样收盘确认、次日开盘卖出
- 止损/追踪止损盘中触发，按止损价成交；跳空穿过止损按开盘价成交
- 单边 5bp 成本；同一股票持仓期间忽略新信号
- 只交易股价 ≥ $5、20 日平均成交额 ≥ $1000 万的股票
- 2019-01-01 之前为样本内，之后为样本外
- 组合模拟：最多 10 只，等权，按净值 1/N 开仓，每日按收盘价盯市

## 输出

- `results/leaderboard.csv`：每个策略的胜率、平均收益、盈亏比、Profit Factor、样本内/外指标、组合年化/回撤/Sharpe、风险提示
- `results/yearly.csv`：逐年胜率
- `results/equity_weekly.csv`：组合周度净值
- `report/index.html`：交互式报告（推荐打法、胜率-盈亏比散点、排行榜、净值与逐年胜率）

## 局限

股票池是**当前**成分股，存在幸存者偏差，绝对收益会被高估；更可靠的是策略之间的相对比较以及相对随机基线的优势。历史表现不代表未来。

## 代码结构

```
backtest/
  universe.py   股票池
  data.py       下载、缓存、宽表面板、合成数据
  indicators.py 指标
  signals.py    策略目录
  engine.py     逐笔模拟（numba）
  portfolio.py  组合模拟、动量轮动
  metrics.py    统计
  run_all.py    主入口
  report.py     报告生成（模板 report_template.html）
tests/          引擎测试
```
