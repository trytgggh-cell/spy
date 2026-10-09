import sys, json, pandas as pd
src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src); df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= "2004-01-01"].reset_index(drop=True)
rows = {}
for d, s in zip(df["date"], df["tickers"]):
    for t in s.split(","):
        r = rows.setdefault(t, [d, d, 0]); r[1] = d; r[2] += 1
last = df["date"].iloc[-1]; cur = set(df["tickers"].iloc[-1].split(","))
print("最近一天", last.date(), "当前成分股", len(cur), "| 2004年以来出现过的代码", len(rows))
removed = {t: r for t, r in rows.items() if t not in cur}
print("已被剔除的代码", len(removed), "占比 %.0f%%" % (100 * len(removed) / len(rows)))
n = df["tickers"].str.count(",") + 1
print("每天成分股数量: 最少", n.min(), "最多", n.max())
# survivors: how many of today's members were in the index on 2008-01-02
d0 = df.loc[df["date"] >= "2008-01-02", "tickers"].iloc[0].split(",")
print("2008-01-02 成分股", len(d0), "其中至今还在指数里的", len(set(d0) & cur), "= %.0f%%" % (100 * len(set(d0) & cur) / len(d0)))
d1 = df.loc[df["date"] >= "2015-01-02", "tickers"].iloc[0].split(",")
print("2015-01-02 成分股", len(d1), "其中至今还在指数里的", len(set(d1) & cur), "= %.0f%%" % (100 * len(set(d1) & cur) / len(d1)))
json.dump({t: [r[0].strftime("%Y-%m-%d"), r[1].strftime("%Y-%m-%d"), r[2]] for t, r in rows.items()}, open(out, "w"))
json.dump(sorted(cur), open(out.replace(".json", "_cur.json"), "w"))
