"""Check the C++ backtests against the pure Python reference (the author's backtest.py).

Usage:
    python check_backtests.py                         # edge cases and synthetic data
    python check_backtests.py path/to/dataset.csv     # also end-to-end on every asset
"""

import sys
import warnings

import numpy as np
import pandas as pd

import creditmc_core as cpp
import reference_backtest as ref
from reference_market_risk import ewma, historical, normal, to_returns

warnings.simplefilter("ignore")  # the reference divides by zero in a degenerate case on purpose


def close(a, b, rtol=1e-9, atol=1e-12):
    return bool(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


failures = 0


def check(label, ok, detail=""):
    global failures
    if not ok:
        failures += 1
    print(f"{'OK  ' if ok else 'FAIL'} {label} {detail}")


# ---- 1. Hit-sequence tests on edge cases and random sequences -------------------------------
sequences = {
    "no hits": [0] * 250,
    "all hits": [1] * 20,
    "single hit": [0] * 100 + [1] + [0] * 149,
    "hit at the start": [1] + [0] * 249,
    "hit at the end": [0] * 249 + [1],
    "clustered": [0] * 50 + [1, 1, 1, 0, 1, 1] + [0] * 100 + [1, 1] + [0] * 92,
    "alternating": [0, 1] * 60,
}
rng = np.random.default_rng(0)
for p in (0.01, 0.05, 0.2):
    for k in range(15):
        sequences[f"random p={p} #{k}"] = list((rng.random(600) < p).astype(int))
# Sequences that start in one state and end in the other.
for p in (0.05, 0.2):
    seq = list((rng.random(600) < p).astype(int))
    seq[0], seq[-1] = 0, 1
    sequences[f"random p={p} 0...1"] = seq
    seq = list((rng.random(600) < p).astype(int))
    seq[0], seq[-1] = 1, 0
    sequences[f"random p={p} 1...0"] = seq

for name, seq in sequences.items():
    h = pd.Series(seq, dtype=int)
    arr = np.array(seq, dtype=int)
    for alpha in (0.01, 0.05):
        for label, fn_ref, fn_cpp in [
            ("kupiec", lambda: ref.kupiec(h, alpha), lambda: cpp.kupiec(arr, alpha)),
            ("conditional", lambda: ref.conditional_coverage(h, alpha),
             lambda: cpp.conditional_coverage(arr, alpha)),
        ]:
            r, c = fn_ref(), fn_cpp()
            check(f"{label:<12} {name:<18} alpha={alpha:<5}",
                  close(r[0], c[0]) and close(r[1], c[1]),
                  f"LR {float(r[0]):.6g} vs {c[0]:.6g}, p {float(r[1]):.6g} vs {c[1]:.6g}")
    r, c = ref.christoffersen_ind(h), cpp.christoffersen_independence(arr)
    check(f"independence {name:<18}", close(r[0], c[0]) and close(r[1], c[1]),
          f"LR {float(r[0]):.6g} vs {c[0]:.6g}, p {float(r[1]):.6g} vs {c[1]:.6g}")
    check(f"traffic      {name:<18}", ref.traffic_light(h) == cpp.traffic_light(arr),
          f"{ref.traffic_light(h)} vs {cpp.traffic_light(arr)}")


# ---- 2. End to end on synthetic data: models -> hits -> every statistic ----------------------
def end_to_end(label, returns, window, alpha, model):
    series = pd.Series(returns, index=pd.bdate_range("2005-01-03", periods=len(returns)))
    py_fn = {"historical": historical, "normal": normal, "ewma": ewma}[model]
    fc = py_fn(series, window=window, alpha=alpha)
    expected = ref.summarize(series, fc, alpha)

    # Same forecasts as plain arrays, run through the C++ backtests.
    r = series.loc[fc.index].to_numpy()
    var, es = fc["VaR"].to_numpy(), fc["ES"].to_numpy()
    h = cpp.hits(r, var)
    got = {
        "n": len(h),
        "hits": int(h.sum()),
        "hit_rate": float(h.mean()),
        "kupiec_p": cpp.kupiec(h, alpha)[1],
        "indep_p": cpp.christoffersen_independence(h)[1],
        "cc_p": cpp.conditional_coverage(h, alpha)[1],
        "es_ratio": cpp.es_ratio(r, var, es),
        "fz0": cpp.fz0(r, var, es, alpha),
    }
    bad = [k for k in got if not close(expected[k], got[k])]
    check(f"end-to-end {model:<10} {label:<22} alpha={alpha:<5}", not bad,
          f"mismatch in {bad}" if bad else f"hits={got['hits']} kupiec_p={got['kupiec_p']:.4f}")


r1 = 0.01 * rng.standard_t(4, 3000)
for model in ("historical", "normal", "ewma"):
    for alpha in (0.05, 0.01):
        end_to_end("Student-t (n=3000)", r1, 250, alpha, model)

if len(sys.argv) > 1:
    print(f"\nDataset: {sys.argv[1]}")
    df = pd.read_csv(sys.argv[1], parse_dates=["Date"], index_col="Date")
    df = df.iloc[:-1]
    for model in ("historical", "normal", "ewma"):
        for asset in df.columns:
            rets = to_returns(df[asset]).to_numpy()
            for alpha in (0.05, 0.01):
                end_to_end(asset, rets, 250, alpha, model)

print(f"\n{'All checks passed.' if failures == 0 else f'{failures} CHECKS FAILED.'}")
sys.exit(0 if failures == 0 else 1)