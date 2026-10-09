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

# ---- renames: current member whose company was added long before its current code appears ----
ren = {}
for t in sp["t"]:
    if t in first and pd.notna(dadd.get(t)) and first[t] > h["date"].iloc[0] + pd.Timedelta(days=5) and (first[t] - dadd[t]).days > 60:
        ren[t] = (dadd[t], first[t])
print("估计为改过代码的现任成分股:", len(ren), "例:", list(ren.items())[:3])

# ---- Part A: how many members are missing, by year ----
close_main = stocks.close; close_rem = rem["close"]
print("\n年份 | 当时成分股 | 主回测数据里有(代码对上) | 改名(数据在,代码不同) | 补入的退市股 | 仍缺 | 缺的比例(主回测) | 缺的比例(补入后)")
for y in (2005, 2008, 2012, 2016, 2020, 2024):
    d = dates[dates >= f"{y}-01-02"][0]; i = np.searchsorted(rdates, np.datetime64(d), side="right") - 1; S = rsets[i]; n = len(S)
    n1 = sum(1 for t in S if t in close_main.columns and pd.notna(close_main.at[d, t]))
    nr = sum(1 for u, (a, b) in ren.items() if a <= d < b and u in close_main.columns and pd.notna(close_main.at[d, u]))
    n2 = sum(1 for t in S if t in close_rem.columns and pd.notna(close_rem.at[d, t]))
    print(f"{y} | {n} | {n1} | {nr} | {n2} | {n - n1 - nr - n2} | {100*(n-n1-nr)/n:.0f}% | {100*(n-n1-nr-n2)/n:.0f}%")

print("\n年份 | 主回测数据里有价格的股票数 | 其中当时还不是成分股(事后才入选)")
for y in (2005, 2008, 2012, 2016, 2020, 2024):
    d = dates[dates >= f"{y}-01-02"][0]; i = np.searchsorted(rdates, np.datetime64(d), side="right") - 1; S = rsets[i]
    have = [t for t in close_main.columns if pd.notna(close_main.at[d, t])]
    notm = [t for t in have if t not in S and not (t in ren and ren[t][0] <= d < ren[t][1])]
    print(f"{y} | {len(have)} | {len(notm)} ({100*len(notm)/len(have):.0f}%)")
if len(sys.argv) > 3: sys.exit()
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

# ---- momentum with eligibility (copy of backtest.portfolio.momentum_rotation + mask) ----
def mom_elig(panel, elig, lookback=252, skip=21, top_n=10, cost=0.0005, start_idx=260):
    c, o = panel.close, panel.open; mom = c.shift(skip) / c.shift(lookback) - 1
    ds = panel.dates; month = ds.to_period("M"); me = np.flatnonzero(month[:-1] != month[1:]); me = me[me >= start_idx]
    eq = pd.Series(np.nan, index=ds); cur = 1.0; rows = []; el = elig.to_numpy(bool) if elig is not None else None; colidx = {t: i for i, t in enumerate(c.columns)}
    for k, s in enumerate(me):
        e = s + 1; x = me[k + 1] + 1 if k + 1 < len(me) else len(ds) - 1
        m = mom.iloc[s].dropna(); m = m[o.iloc[e][m.index].notna()]
        if elig is not None: m = m[[el[s, colidx[t]] for t in m.index]]
        picks = m.nlargest(top_n).index
        if len(picks) == 0: eq.iloc[e:x + 1] = cur; continue
        ep = o.iloc[e][picks] * (1 + cost); path = c.iloc[e:x + 1][picks].ffill() / ep
        xp = o.iloc[x][picks]; xp = xp.fillna(c.iloc[:x + 1][picks].ffill().iloc[-1]) * (1 - cost); path.iloc[-1] = xp / ep
        port = path.mean(axis=1) * cur; eq.iloc[e:x + 1] = port.to_numpy(); cur = float(port.iloc[-1])
        for t in picks: rows.append(dict(ticker=t, entry_date=ds[e], ret=xp[t] / ep[t]))
    return eq.ffill().fillna(1.0), pd.DataFrame(rows)

res = {}
for name, panel, el in (("A 今天的成分股(原回测)", stocks, None), ("B 今天的成分股,入选后才能买", stocks, E1.loc[:, stocks.close.columns]), ("C 当时的成分股,补入能找回的退市股", comb, E2)):
    eq, hh = mom_elig(panel, el); eq = eq[eq.index >= eq.index[(eq.diff().abs() > 0).to_numpy().argmax()]]
    st = metrics.equity_stats(eq / eq.iloc[0], ""); res[name] = st
    extra = ""
    if name.startswith("C"):
        isrem = hh["ticker"].isin(rem["close"].columns); extra = f" | 选中的退市股笔数 {int(isrem.sum())} ({100*isrem.mean():.1f}%), 平均月收益 {100*hh.loc[isrem,'ret'].mean():+.1f}% vs 现任 {100*hh.loc[~isrem,'ret'].mean():+.1f}%"
    print(f"{name}: 年化 {100*st['cagr']:.1f}% | 10000→{10000*st['final']:,.0f} | 最大回撤 {100*st['max_dd']:.0f}% | 夏普 {st['sharpe']:.2f}{extra}")
