"""Check the C++ rolling VaR/ES models (historical, normal, EWMA) against the pure Python
reference, which is the author's own models.py code.

Usage:
    python check_var_models.py                         # synthetic data only
    python check_var_models.py path/to/dataset.csv     # also every asset in your dataset
"""

import sys
import time

import numpy as np
import pandas as pd

import creditmc_core as cpp
from reference_market_risk import ewma, historical, normal, to_returns

TOLERANCE = 1e-12

# name -> (python reference, C++ version); both take (returns, window, alpha).
MODELS = {
    "historical": (lambda s, w, a: historical(s, window=w, alpha=a),
                   lambda r, w, a: cpp.historical_var_es(r, w, a)),
    "normal": (lambda s, w, a: normal(s, window=w, alpha=a),
               lambda r, w, a: cpp.normal_var_es(r, w, a)),
    "ewma": (lambda s, w, a: ewma(s, window=w, alpha=a),
             lambda r, w, a: cpp.ewma_var_es(r, w, a, 0.94)),
}


def compare(model, label, returns, window, alpha):
    py_fn, cpp_fn = MODELS[model]
    returns = np.asarray(returns, dtype=float)
    series = pd.Series(returns, index=pd.bdate_range("2005-01-03", periods=len(returns)))

    t0 = time.perf_counter()
    ref = py_fn(series, window, alpha)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    var, es = cpp_fn(returns, window, alpha)
    t_cpp = time.perf_counter() - t0

    same_length = len(var) == len(ref)
    d_var = np.max(np.abs(var - ref["VaR"].to_numpy())) if same_length else np.inf
    d_es = np.max(np.abs(es - ref["ES"].to_numpy())) if same_length else np.inf
    ok = same_length and d_var < TOLERANCE and d_es < TOLERANCE
    print(f"{'OK  ' if ok else 'FAIL'} {model:<10} {label:<24} window={window:<4} alpha={alpha:<5} "
          f"max|dVaR|={d_var:.1e} max|dES|={d_es:.1e}  python {t_py:7.4f}s  C++ {t_cpp:7.4f}s")
    return ok


rng = np.random.default_rng(0)
cases = {
    "Student-t (n=3000)": 0.01 * rng.standard_t(4, 3000),
    "rounded, ties (n=3000)": np.round(0.01 * rng.standard_t(4, 3000), 4),
}

all_ok = True
for model in MODELS:
    for label, r in cases.items():
        for window, alpha in [(250, 0.05), (250, 0.01), (60, 0.05), (500, 0.01)]:
            all_ok &= compare(model, label, r, window, alpha)

if len(sys.argv) > 1:
    print(f"\nDataset: {sys.argv[1]}")
    df = pd.read_csv(sys.argv[1], parse_dates=["Date"], index_col="Date")
    df = df.iloc[:-1]
    for model in MODELS:
        for asset in df.columns:
            r = to_returns(df[asset])
            for alpha in (0.05, 0.01):
                all_ok &= compare(model, asset, r.to_numpy(), 250, alpha)

print("\nAll checks passed." if all_ok else "\nSOME CHECKS FAILED.")
sys.exit(0 if all_ok else 1)