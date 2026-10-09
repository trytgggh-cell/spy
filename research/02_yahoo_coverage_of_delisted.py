import sys, json, time, warnings
import pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")
rng = json.load(open(sys.argv[1])); cur = set(json.load(open(sys.argv[2])))
out = sys.argv[3]
removed = sorted(t for t in rng if t not in cur)
ymap = {t: t.replace(".", "-") for t in removed}
frames, res = [], []
B = 40
for i in range(0, len(removed), B):
    chunk = removed[i:i + B]
    try:
        raw = yf.download([ymap[t] for t in chunk], start="2003-06-01", auto_adjust=True, progress=False, group_by="ticker", threads=True)
    except Exception as e:
        print("batch err", e); continue
    for t in chunk:
        y = ymap[t]
        if y not in raw.columns.get_level_values(0): continue
        d = raw[y].dropna(how="all")
        if d.empty: continue
        d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].reset_index()
        d.columns = ["date"] + list(d.columns[1:]); d["ticker"] = t
        frames.append(d)
    print(min(i + B, len(removed)), "/", len(removed), "with data so far:", len(frames), flush=True)
allp = pd.concat(frames, ignore_index=True)
allp["date"] = pd.to_datetime(allp["date"]).dt.tz_localize(None)
allp.to_parquet(out + "/removed_prices.parquet", index=False)
g = allp.groupby("ticker")["date"].agg(["min", "max", "size"])
rows = []
for t in removed:
    fi, la, n = pd.Timestamp(rng[t][0]), pd.Timestamp(rng[t][1]), rng[t][2]
    if t in g.index:
        a, b, k = g.loc[t]
        # covers the membership window? (allow 45 days slack at each end; index start clipped at 2004-01)
        cover_start = a <= max(fi, pd.Timestamp("2004-01-02")) + pd.Timedelta(days=45)
        cover_end = b >= la - pd.Timedelta(days=45)
        rows.append(dict(ticker=t, has_data=True, first=a, last=b, in_first=fi, in_last=la, covers=bool(cover_start and cover_end), partial=bool(b >= la - pd.Timedelta(days=45))))
    else:
        rows.append(dict(ticker=t, has_data=False, in_first=fi, in_last=la, covers=False, partial=False))
r = pd.DataFrame(rows); r.to_csv(out + "/coverage.csv", index=False)
print("被剔除代码", len(r), "| Yahoo 有数据", int(r.has_data.sum()), "| 数据完整覆盖其在指数里的时期", int(r.covers.sum()))
print("按在指数里的天数加权: 完整覆盖占比 %.0f%%" % (100 * sum(rng[t][2] for t in r[r.covers].ticker) / sum(rng[t][2] for t in r.ticker)))
