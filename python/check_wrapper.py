"""Check that fast_risk (the C++-backed wrappers) behaves exactly like the author's original
pandas code: same values, same index, same column names, same return types.

Usage:
    python check_wrapper.py                         # synthetic data
    python check_wrapper.py path/to/dataset.csv     # also every asset in your dataset
"""

import sys
import warnings

import numpy as np
import pandas as pd

import fast_risk as fast
import reference_backtest as ref_bt
import reference_market_risk as ref_mr

warnings.simplefilter("ignore")

failures = 0


def check(label, ok, detail=""):
    global failures
    if not ok:
        failures += 1
    print(f"{'OK  ' if ok else 'FAIL'} {label} {detail}")


def frames_equal(a, b):
    try:
        # check_freq=False: pandas keeps a "BusinessDay" frequency tag on sliced indexes but not on
        # indexes rebuilt row by row; the dates themselves are identical.
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12, rtol=0, check_freq=False)
        return True
    except AssertionError as exc:
        print("     ", str(exc).splitlines()[0])
        return False


def close(a, b):
    return bool(np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True))


def models_and_backtests(label, returns, window, alpha):
    pairs = [
        ("historical", ref_mr.historical(returns, window=window, alpha=alpha),
         fast.historical(returns, window=window, alpha=alpha)),
        ("normal", ref_mr.normal(returns, window=window, alpha=alpha),
         fast.normal(returns, window=window, alpha=alpha)),
        ("ewma", ref_mr.ewma(returns, window=window, alpha=alpha),
         fast.ewma(returns, window=window, alpha=alpha)),
    ]
    for model, expected, got in pairs:
        check(f"model    {model:<10} {label:<20} alpha={alpha:<5}", frames_equal(got, expected))

        # Backtests on the reference forecasts, original code against the wrapper.
        e = ref_bt.summarize(returns, expected, alpha)
        g = fast.summarize(returns, expected, alpha)
        bad = [k for k in e if not close(e[k], g[k])]
        check(f"summarize {model:<10} {label:<20} alpha={alpha:<5}", not bad,
              f"mismatch in {bad}" if bad else f"hits={g['hits']}")

        ev = ref_bt.summarize_var(returns, expected["VaR"], alpha)
        gv = fast.summarize_var(returns, expected["VaR"], alpha)
        bad = [k for k in ev if not close(ev[k], gv[k])]
        check(f"summ_var  {model:<10} {label:<20} alpha={alpha:<5}", not bad,
              f"mismatch in {bad}" if bad else "")

        h_ref = ref_bt.hits(returns, expected["VaR"])
        h_fast = fast.hits(returns, expected["VaR"])
        check(f"hits      {model:<10} {label:<20} alpha={alpha:<5}",
              h_ref.reset_index(drop=True).equals(h_fast.reset_index(drop=True))
              and h_ref.index.equals(h_fast.index))
        check(f"traffic   {model:<10} {label:<20} alpha={alpha:<5}",
              ref_bt.traffic_light(h_ref) == fast.traffic_light(h_fast))
        r1, r2 = ref_bt.es_ratio_test(returns, expected["VaR"], expected["ES"]), \
            fast.es_ratio_test(returns, expected["VaR"], expected["ES"])
        check(f"es_ratio  {model:<10} {label:<20} alpha={alpha:<5}",
              close(r1[0], r2[0]) and close(r1[1], r2[1]),
              f"mean {r1[0]:.6g} vs {r2[0]:.6g}, p {r1[1]:.6g} vs {r2[1]:.6g}")


rng = np.random.default_rng(0)
index = pd.bdate_range("2005-01-03", periods=3000, name="Date")
synthetic = pd.Series(0.01 * rng.standard_t(4, 3000), index=index)
rounded = pd.Series(np.round(synthetic.to_numpy(), 3), index=index)

for label, series in (("Student-t", synthetic), ("rounded (ties)", rounded)):
    for alpha in (0.05, 0.01):
        models_and_backtests(label, series, 250, alpha)

# ---- panel: many assets with different histories, in parallel ----------------------------------
panel = pd.DataFrame({
    "A": synthetic,
    "B": synthetic.shift(-1) * 0.8,
    "C": pd.Series(0.02 * rng.standard_t(5, 3000), index=index),
})
panel.loc[panel.index[:700], "B"] = np.nan   # B has a shorter history
panel.loc[panel.index[:200], "C"] = np.nan   # so does C, differently
panel = panel.dropna(how="all")

for model, reference in (("historical", ref_mr.historical), ("normal", ref_mr.normal),
                         ("ewma", ref_mr.ewma)):
    for threads in (1, 4):
        got = fast.var_es_panel(panel, model=model, window=250, alpha=0.01, n_threads=threads)
        ok = list(got) == list(panel.columns) and all(
            frames_equal(got[c], reference(panel[c].dropna(), window=250, alpha=0.01))
            for c in panel.columns)
        check(f"panel    {model:<10} threads={threads}", ok)

try:
    fast.var_es_panel(panel, model="garch")
    check("panel rejects an unknown model", False)
except ValueError:
    check("panel rejects an unknown model", True)

# ---- optional: the user's real dataset ---------------------------------------------------------
if len(sys.argv) > 1:
    print(f"\nDataset: {sys.argv[1]}")
    df = pd.read_csv(sys.argv[1], parse_dates=["Date"], index_col="Date").iloc[:-1]
    for asset in df.columns:
        r = ref_mr.to_returns(df[asset])
        check(f"to_returns {asset:<9}", r.equals(fast.to_returns(df[asset])))
        for alpha in (0.05, 0.01):
            models_and_backtests(asset, r, 250, alpha)

print(f"\n{'All checks passed.' if failures == 0 else f'{failures} CHECKS FAILED.'}")
sys.exit(0 if failures == 0 else 1)