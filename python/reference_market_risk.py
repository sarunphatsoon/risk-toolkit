"""Reference implementations copied from the author's models.py, without the `arch` import,
so they can be used in tests and CI. The C++ code must reproduce these results."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def to_returns(prices):
    return np.log(prices.dropna()).diff().dropna()


def historical(returns, window=250, alpha=0.05):
    rows = []
    for i in range(window, len(returns)):
        past = returns.iloc[i - window:i]
        q = np.quantile(past, alpha)
        tail = past[past <= q]
        rows.append((returns.index[i], -q, -tail.mean()))
    return pd.DataFrame(rows, columns=["Date", "VaR", "ES"]).set_index("Date")


def normal(returns, window=250, alpha=0.05):
    mu = returns.rolling(window).mean().shift(1)
    sd = returns.rolling(window).std().shift(1)
    z = norm.ppf(alpha)
    var = -(mu + sd * z)
    es = -mu + sd * norm.pdf(z) / alpha
    return pd.DataFrame({"VaR": var, "ES": es}).dropna()


def ewma(returns, window=250, alpha=0.05, lam=0.94):
    v = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean().shift(1)
    sd = np.sqrt(v)
    z = norm.ppf(alpha)
    out = pd.DataFrame({"VaR": -sd * z,
                        "ES": sd * norm.pdf(z) / alpha})
    return out.iloc[window:]