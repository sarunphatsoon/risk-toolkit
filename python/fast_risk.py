"""Drop-in, pandas-friendly wrappers around the C++ risk kernels (creditmc_core).

The function names, arguments and return types match the author's models.py and backtest.py,
so a script can switch to the C++ versions by changing its imports:

    from models import student_t, garch          # still pure Python (they need the arch package)
    from fast_risk import historical, normal, ewma
    from fast_risk import summarize, summarize_var, kupiec, traffic_light, ...

Differences from the originals:
  * The C++ code rejects non-finite returns and invalid arguments with a ValueError, where the
    pandas versions would quietly produce NaN. Drop missing values first, as to_returns does.
  * var_es_panel() is new: it runs many assets in parallel across threads.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

import creditmc_core as _core


def _values(x):
    """Any array-like (Series, list, ndarray) as a contiguous float64 array."""
    return np.ascontiguousarray(np.asarray(x, dtype=float))


def _frame(index, var, es):
    return pd.DataFrame({"VaR": var, "ES": es}, index=index)


# ---------------------------------------------------------------------------------------------
# Models (same signatures as models.py)
# ---------------------------------------------------------------------------------------------

def to_returns(prices):
    return np.log(prices.dropna()).diff().dropna()


def historical(returns, window=250, alpha=0.05):
    var, es = _core.historical_var_es(_values(returns), window, alpha)
    out = _frame(returns.index[window:], var, es)
    out.index.name = "Date"
    return out


def normal(returns, window=250, alpha=0.05):
    var, es = _core.normal_var_es(_values(returns), window, alpha)
    return _frame(returns.index[window:], var, es)


def ewma(returns, window=250, alpha=0.05, lam=0.94):
    var, es = _core.ewma_var_es(_values(returns), window, alpha, lam)
    return _frame(returns.index[window:], var, es)


def var_es_panel(returns, model="historical", window=250, alpha=0.05, lam=0.94, n_threads=0):
    """VaR/ES for every column of a DataFrame of returns, in parallel across assets.

    Missing values are dropped column by column, so assets may have different histories.
    Returns {asset name: DataFrame with columns VaR and ES}, identical to calling the
    single-asset function on each column.
    """
    series = {col: returns[col].dropna() for col in returns.columns}
    arrays = [_values(s) for s in series.values()]
    if model == "historical":
        out = _core.historical_var_es_batch(arrays, window, alpha, n_threads)
    elif model == "normal":
        out = _core.normal_var_es_batch(arrays, window, alpha, n_threads)
    elif model == "ewma":
        out = _core.ewma_var_es_batch(arrays, window, alpha, lam, n_threads)
    else:
        raise ValueError("model must be 'historical', 'normal' or 'ewma'")

    result = {}
    for (col, s), (var, es) in zip(series.items(), out):
        frame = _frame(s.index[window:], var, es)
        if model == "historical":
            frame.index.name = "Date"
        result[col] = frame
    return result


# ---------------------------------------------------------------------------------------------
# Backtests (same signatures as backtest.py)
# ---------------------------------------------------------------------------------------------

def _hit_array(h):
    return np.ascontiguousarray(np.asarray(h, dtype=int))


def hits(returns, var):
    r = returns.loc[var.index]
    h = _core.hits(_values(r), _values(var))
    return pd.Series(h.astype(int), index=var.index)


def kupiec(h, alpha):
    return _core.kupiec(_hit_array(h), alpha)


def christoffersen_ind(h):
    return _core.christoffersen_independence(_hit_array(h))


def conditional_coverage(h, alpha):
    return _core.conditional_coverage(_hit_array(h), alpha)


def es_ratio_test(returns, var, es):
    r = returns.loc[var.index]
    mean_ratio = _core.es_ratio(_values(r), _values(var), _values(es))
    if np.isnan(mean_ratio):
        return np.nan, np.nan
    exceeded = _values(r) < -_values(var)
    ratio = (-_values(r)[exceeded]) / _values(es)[exceeded]
    return mean_ratio, ttest_1samp(ratio, 1.0).pvalue


def traffic_light(h, window=250):
    return _core.traffic_light(_hit_array(h), window)


def fz0(returns, var, es, alpha):
    r = returns.loc[var.index]
    return _core.fz0(_values(r), _values(var), _values(es), alpha)


def summarize(returns, fc, alpha):
    h = hits(returns, fc["VaR"])
    return {
        "n": len(h),
        "hits": int(h.sum()),
        "expected": round(alpha * len(h), 1),
        "hit_rate": h.mean(),
        "kupiec_p": kupiec(h, alpha)[1],
        "indep_p": christoffersen_ind(h)[1],
        "cc_p": conditional_coverage(h, alpha)[1],
        "es_ratio": es_ratio_test(returns, fc["VaR"], fc["ES"])[0],
        "fz0": fz0(returns, fc["VaR"], fc["ES"], alpha),
    }


def summarize_var(returns, var, alpha):
    h = hits(returns, var)
    return {
        "n": len(h),
        "hits": int(h.sum()),
        "expected": round(alpha * len(h), 1),
        "hit_rate": h.mean(),
        "kupiec_p": kupiec(h, alpha)[1],
        "indep_p": christoffersen_ind(h)[1],
        "cc_p": conditional_coverage(h, alpha)[1],
    }