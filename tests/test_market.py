import numpy as np
import pandas as pd

from backtest.market import dollar_volume_rank_mask, filter_common_stocks


def test_filter_keeps_plain_common_stock_only():
    df = pd.DataFrame({
        "Symbol": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
        "Security Name": ["Alpha Inc. - Common Stock", "Beta Corp. - Warrant", "Gamma ETF - Common Stock",
                          "Delta Inc. - 6.5% Series A Preferred Stock", "Eps Corp. - Class A Common Stock",
                          "Zeta Acquisition Corp. - Common Stock", "Eta Ltd. - Ordinary Shares"],
        "Test Issue": ["N", "N", "N", "N", "N", "N", "Y"],
        "ETF": ["N", "N", "Y", "N", "N", "N", "N"],
    })
    assert filter_common_stocks(df, "Symbol") == ["AAA", "EEE"]


def test_dollar_volume_rank_mask_is_point_in_time_and_price_gated():
    idx = pd.bdate_range("2020-01-01", periods=30)
    close = pd.DataFrame({"BIG": 50.0, "MID": 20.0, "SMALL": 10.0, "PENNY": 2.0}, index=idx)
    vol = pd.DataFrame({"BIG": 1e6, "MID": 1e6, "SMALL": 1e6, "PENNY": 1e9}, index=idx)
    m = dollar_volume_rank_mask(close, vol, n=2, window=5, min_price=5.0)
    assert not m.iloc[:4].to_numpy().any()                  # needs a full window
    last = m.iloc[-1]
    assert last["BIG"] and last["MID"] and not last["SMALL"]
    assert not last["PENNY"]                                # huge volume but price < $5
    # a stock that only becomes liquid later is not in the universe before that date
    vol2 = vol.copy(); vol2.loc[idx[15]:, "SMALL"] = 1e8
    m2 = dollar_volume_rank_mask(close, vol2, n=2, window=5, min_price=5.0)
    assert not m2["SMALL"].iloc[:15].any() and m2["SMALL"].iloc[-1]
