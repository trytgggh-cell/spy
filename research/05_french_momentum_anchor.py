import sys, re, io, numpy as np, pandas as pd
txt = open(sys.argv[1], encoding="latin-1").read().splitlines()
def section(title):
    i = next(k for k, l in enumerate(txt) if l.strip().startswith(title))
    hdr = txt[i + 1]; cols = [c.strip() for c in hdr.split(",")][1:]
    rows = []
    for l in txt[i + 2:]:
        p = [x.strip() for x in l.split(",")]
        if not re.fullmatch(r"\d{6}", p[0]): break
        rows.append([p[0]] + [float(x) for x in p[1:]])
    df = pd.DataFrame(rows, columns=["ym"] + cols); df.index = pd.PeriodIndex(df.pop("ym"), freq="M")
    return df / 100
vw = section("Value Weight Returns -- Monthly"); ew = section("Average Equal Weighted Returns -- Monthly")
print(vw.columns.tolist(), vw.index[0], vw.index[-1])
def stats(r):
    w = (1 + r).cumprod(); yrs = len(r) / 12
    dd = (w / w.cummax() - 1).min(); y = (1 + r).groupby(r.index.year).prod() - 1
    return dict(cagr=w.iloc[-1] ** (1 / yrs) - 1, mdd=dd, worst_year=y.min(), worst_y=int(y.idxmin()), final=10000 * w.iloc[-1], vol=r.std() * np.sqrt(12), losing=int((y < 0).sum()), n=len(y), monthly_win=(r > 0).mean(), worst_m=r.min(), best_m=r.max())
for per, a, b in (("2005-02 起（和我们的回测同一区间）", "2005-02", None), ("1927 起（全部历史）", None, None), ("2010 起", "2010-01", None)):
    print("\n==", per)
    for nm, d in (("价值加权", vw), ("等权", ew)):
        sub = d.loc[a:b] if a else d
        hi, lo = sub.iloc[:, -1], sub.iloc[:, 0]; allm = sub.mean(axis=1)
        for lab, r in (("最强一档(Hi PRIOR)", hi), ("最弱一档(Lo PRIOR)", lo), ("十档平均(大盘代理)", allm)):
            s = stats(r); print("%s %-14s 年化 %5.1f%% | 10000→%11s | 最大回撤 %4.0f%% | 最差年 %+4.0f%%(%d) | 亏钱年 %d/%d | 月胜率 %2.0f%% 最差月 %+4.0f%%" % (nm, lab, 100 * s["cagr"], f"{s['final']:,.0f}", 100 * s["mdd"], 100 * s["worst_year"], s["worst_y"], s["losing"], s["n"], 100 * s["monthly_win"], 100 * s["worst_m"]))
