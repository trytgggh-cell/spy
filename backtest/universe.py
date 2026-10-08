"""Stock universe: S&P 500 (fetched) + Nasdaq-100 (hard-coded), plus benchmarks."""
from __future__ import annotations

import io
import urllib.request

import pandas as pd

SP500_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)

# Nasdaq-100 members (2025-2026). Overlap with the S&P 500 is removed later.
NASDAQ100 = """
AAPL ABNB ADBE ADI ADP ADSK AEP AMAT AMD AMGN AMZN APP ARM ASML AVGO AXON AZN
BIIB BKNG BKR CCEP CDNS CDW CEG CHTR CMCSA COST CPRT CRWD CSCO CSGP CSX CTAS
CTSH DASH DDOG DXCM EA EXC FANG FAST FTNT GEHC GFS GILD GOOG GOOGL HON IDXX
INTC INTU ISRG KDP KHC KLAC LIN LRCX LULU MAR MCHP MDLZ MELI META MNST MRVL
MSFT MSTR MU NFLX NVDA NXPI ODFL ON ORLY PANW PAYX PCAR PDD PEP PLTR PYPL QCOM
REGN ROP ROST SBUX SHOP SNPS TEAM TMUS TRI TSLA TTD TTWO TXN VRSK VRTX WBD
WDAY XEL ZS
""".split()

BENCHMARKS = ["SPY", "QQQ"]


def _yahoo_symbol(sym: str) -> str:
    return sym.strip().upper().replace(".", "-")


def fetch_sp500() -> pd.DataFrame:
    req = urllib.request.Request(SP500_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    df["Symbol"] = df["Symbol"].map(_yahoo_symbol)
    return df[["Symbol", "Security", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"}
    )


def get_universe() -> pd.DataFrame:
    """Return DataFrame[ticker, name, sector, source] of unique stock tickers."""
    sp = fetch_sp500()
    sp["source"] = "SP500"
    extra = sorted(set(map(_yahoo_symbol, NASDAQ100)) - set(sp["ticker"]))
    ndx = pd.DataFrame(
        {"ticker": extra, "name": extra, "sector": "Unknown", "source": "NDX100"}
    )
    return pd.concat([sp, ndx], ignore_index=True).drop_duplicates("ticker")


if __name__ == "__main__":
    u = get_universe()
    print(u["source"].value_counts())
    print(len(u), "tickers")
