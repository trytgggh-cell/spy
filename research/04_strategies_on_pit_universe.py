import sys, io, json, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from backtest import metrics
from backtest.data import load_or_download, Panel
from backtest.portfolio import momentum_rotation as mom_orig
hist, out_dir = sys.argv[1], sys.argv[2]
Y = lambda s: s.replace(".", "-")

# ---- PIT membership rows (2004+) ----
h = pd.read_csv(hist); h["date"] = pd.to_datetime(h["date"]); h = h[h["date"] >= "2004-01-01"].reset_index(drop=True)
rdates = h["date"].to_numpy(); rsets = [set(Y(t) for t in s.split(",")) for s in h["tickers"]]
first, last = {}, {}
for d, s in zip(h["date"], rsets):
    for t in s:
        first.setdefault(t, d); last[t] = d
# ---- current members + company-level date added ----
url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
sp = pd.read_csv(io.StringIO(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})).read().decode()))
sp["t"] = sp["Symbol"].map(Y); sp["added"] = pd.to_datetime(sp["Date added"], errors="coerce")
dadd = dict(zip(sp["t"], sp["added"]))
# ---- panel ----
full = load_or_download([]); names = [t for t in full.tickers if t not in ("SPY", "QQQ")]
stocks = full.subset(names); dates = stocks.dates; cols = list(stocks.close.columns)
cov = pd.read_csv(out_dir + "/coverage.csv"); good = [Y(t) for t in cov.loc[cov["covers"], "ticker"]]
overlap = [t for t in good if t in set(stocks.close.columns)]
print("与现有数据代码重复(同一代码被不同公司使用,剔除):", overlap)
good = [t for t in good if t not in set(overlap)]
rp = pd.read_parquet(out_dir + "/removed_prices.parquet"); rp["ticker"] = rp["ticker"].map(Y); rp = rp[rp["ticker"].isin(good)]
def wide(f): return rp.pivot_table(index="date", columns="ticker", values=f).reindex(dates)
rem = {f: wide(f) for f in ("open", "high", "low", "close", "volume")}
comb = Panel(*[pd.concat([getattr(stocks, f), rem[f]], axis=1) for f in ("open", "high", "low", "close", "volume")])
print("补入的退市股（Yahoo 数据完整覆盖其在指数里的时期）:", len(rem["close"].columns))

# ---- eligibility masks ----
def elig_cur():
    e = pd.DataFrame(False, index=dates, columns=comb.close.columns)
    for t in stocks.close.columns:
        a = dadd.get(t)
        if pd.notna(a): e[t] = dates >= a
    return e
E1 = elig_cur()
E2 = E1.copy()
for t in rem["close"].columns:
    pres = pd.Series([t in s for s in rsets], index=pd.DatetimeIndex(rdates))
    E2[t] = pres.reindex(dates, method="ffill").fillna(False).to_numpy(bool)


from backtest.signals import Ctx, build_strategies
from backtest.engine import run_trades
from backtest.portfolio import simulate_slots
S = {x.name: x for x in build_strategies()}
pick = ["5日内从高点回撤>8% 上升趋势 | 收盘>SMA5出场", "RSI2<5 上升趋势 | 收盘>SMA5出场 | 最长10天", "跌破布林下轨 上升趋势 | 回到中轨出场",
        "50/200均线金叉 | 死叉出场", "55日新高 | 6ATR追踪止损", "52周新高 | 8ATR追踪止损", "20日新高突破(海龟) | 3ATR追踪止损"]
scols = list(stocks.close.columns)
variants = [("A 今天的成分股", stocks, None), ("B 入选后才能买", stocks, E1[scols]), ("C 当时成分股+补回退市股", comb, E2)]
ctxs = {}
rows = []
for nm in pick:
    st_ = S[nm]
    for vn, panel, mask in variants:
        key = id(panel)
        if key not in ctxs: ctxs[key] = Ctx(panel)
        ctx = ctxs[key]
        ent = st_.entry(ctx)
        if mask is not None: ent = ent & mask.reindex(index=ent.index, columns=ent.columns).fillna(False)
        ex = st_.exit(ctx) if st_.exit else None; sc = st_.score(ctx) if st_.score else None
        tr = run_trades(panel, ent, st_.rule, ex, ctx.atr14 if st_.use_atr else None, sc, 0.0005)
        eq, _ = simulate_slots(panel, tr, 10)
        e = metrics.equity_stats(eq, ""); o = metrics.equity_stats(eq[eq.index >= metrics.OOS_START], "")
        r = tr["ret"].to_numpy()
        rows.append(dict(strategy=nm, variant=vn, trades=len(r), win=(r > 0).mean(), avg=r.mean(), cagr=e["cagr"], sharpe=e["sharpe"], mdd=e["max_dd"], oos_cagr=o["cagr"], final=10000 * e["final"], worst_year=e["worst_year"]))
        print(nm[:22], vn, f"年化 {100*e['cagr']:.1f}% 夏普 {e['sharpe']:.2f} 回撤 {100*e['max_dd']:.0f}% 2019后年化 {100*o['cagr']:.1f}% 胜率 {100*(r>0).mean():.0f}% 笔数 {len(r)}", flush=True)
pd.DataFrame(rows).to_csv(out_dir + "/pit_strategies.csv", index=False)
