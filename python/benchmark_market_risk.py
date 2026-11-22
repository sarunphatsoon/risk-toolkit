"""Benchmark rolling historical VaR/ES on a panel of assets.

Compares the author's pandas loop, a vectorised NumPy version, and the C++ batch kernel
(one thread and all threads). Slow baselines are timed on a small sample of assets and
extrapolated to the full panel; the C++ runs process the whole panel.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import creditmc_core as cpp
from reference_market_risk import historical

N_ASSETS = 2000
N_DAYS = 5000
WINDOW = 250
ALPHA = 0.01
N_REPEATS = 3
PANDAS_SAMPLE = 3    # assets timed for the pandas loop, then extrapolated
NUMPY_SAMPLE = 25    # assets timed for the vectorised NumPy version, then extrapolated


def historical_numpy(x, window, alpha):
    """Vectorised rolling historical VaR/ES: sort every window at once with NumPy."""
    windows = sliding_window_view(x, window)[:-1]      # window k covers days k .. k+window-1
    q = np.quantile(windows, alpha, axis=1)
    in_tail = windows <= q[:, None]
    es = -(np.where(in_tail, windows, 0.0).sum(axis=1) / in_tail.sum(axis=1))
    return -q, es


rng = np.random.default_rng(0)
panel = [0.01 * rng.standard_t(4, N_DAYS) for _ in range(N_ASSETS)]
print(f"Panel: {N_ASSETS} assets x {N_DAYS} days, window {WINDOW}, alpha {ALPHA}\n")

# ---- 1. Correctness: the three implementations must agree --------------------------------------
check = cpp.historical_var_es_batch(panel[:5], WINDOW, ALPHA, 1)
worst = 0.0
for k in range(5):
    series = pd.Series(panel[k], index=pd.bdate_range("2005-01-03", periods=N_DAYS))
    ref = historical(series, window=WINDOW, alpha=ALPHA)
    var_np, es_np = historical_numpy(panel[k], WINDOW, ALPHA)
    var_c, es_c = check[k]
    worst = max(worst,
                np.abs(var_c - ref["VaR"].to_numpy()).max(), np.abs(es_c - ref["ES"].to_numpy()).max(),
                np.abs(var_c - var_np).max(), np.abs(es_c - es_np).max())
print(f"Largest difference between C++, pandas loop and vectorised NumPy (5 assets): {worst:.1e}")
if worst > 1e-12:
    sys.exit("Implementations disagree; refusing to benchmark.")

# ---- 2. Slow baselines on a sample, extrapolated -----------------------------------------------
t0 = time.perf_counter()
for k in range(PANDAS_SAMPLE):
    series = pd.Series(panel[k], index=pd.bdate_range("2005-01-03", periods=N_DAYS))
    historical(series, window=WINDOW, alpha=ALPHA)
pandas_total = (time.perf_counter() - t0) / PANDAS_SAMPLE * N_ASSETS

t0 = time.perf_counter()
for k in range(NUMPY_SAMPLE):
    historical_numpy(panel[k], WINDOW, ALPHA)
numpy_total = (time.perf_counter() - t0) / NUMPY_SAMPLE * N_ASSETS

# ---- 3. C++ batch kernel on the full panel, by thread count ------------------------------------
max_cpu = os.cpu_count() or 1
thread_counts = sorted({t for t in (1, 2, 4, 8, 16) if t <= max_cpu} | {max_cpu})
cpp.historical_var_es_batch(panel[:4], WINDOW, ALPHA, 1)  # warm-up

cpp_time = []
reference = None
for n_threads in thread_counts:
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        out = cpp.historical_var_es_batch(panel, WINDOW, ALPHA, n_threads)
        times.append(time.perf_counter() - t0)
    cpp_time.append(float(np.median(times)))
    if reference is None:
        reference = out
    identical = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                    for a, b in zip(reference, out))
    print(f"C++ batch, {n_threads:2d} threads: median {cpp_time[-1]:7.3f}s   identical to 1 thread: {identical}")

# ---- 4. Report ---------------------------------------------------------------------------------
rows = [
    (f"pandas loop (est. from {PANDAS_SAMPLE} assets)", pandas_total),
    (f"vectorised NumPy (est. from {NUMPY_SAMPLE} assets)", numpy_total),
    ("C++ batch, 1 thread", cpp_time[0]),
    (f"C++ batch, {thread_counts[-1]} threads", cpp_time[-1]),
]
print(f"\n{'method':<44}{'panel time':>12}{'per asset':>12}{'vs pandas':>11}{'vs NumPy':>10}")
for name, total in rows:
    print(f"{name:<44}{total:>11.2f}s{total / N_ASSETS * 1000:>10.1f}ms"
          f"{pandas_total / total:>10.1f}x{numpy_total / total:>9.1f}x")

counts = np.array(thread_counts, dtype=float)
speedup = cpp_time[0] / np.array(cpp_time)
print(f"\n{'threads':>8}{'time (s)':>10}{'speedup':>9}{'efficiency':>12}")
for n, t, s in zip(thread_counts, cpp_time, speedup):
    print(f"{n:>8}{t:>10.3f}{s:>8.2f}x{s / n * 100:>11.0f}%")

# ---- 5. Plot -----------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

labels = [r[0] for r in rows][::-1]
values = [r[1] for r in rows][::-1]
bars = ax1.barh(labels, values, color=["tab:green", "tab:green", "tab:blue", "tab:red"][::-1])
ax1.set_xscale("log")
ax1.set_xlabel("Time to process the whole panel (s, log scale)")
ax1.set_title(f"Rolling historical VaR/ES: {N_ASSETS} assets x {N_DAYS} days")
for bar, v in zip(bars, values):
    ax1.text(v * 1.08, bar.get_y() + bar.get_height() / 2, f"{v:.2f}s", va="center")
ax1.set_xlim(right=max(values) * 6)
ax1.grid(alpha=0.3, axis="x")

ax2.plot(counts, counts, linestyle="--", color="grey", label="Ideal (linear)")
ax2.plot(counts, speedup, marker="o", label="C++ batch")
ax2.set_xlabel("Threads")
ax2.set_ylabel("Speedup vs 1 thread")
ax2.set_title("Thread scaling on the panel")
ax2.set_xticks(counts)
ax2.grid(alpha=0.3)
ax2.legend()

fig.tight_layout()
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "market_risk_benchmark.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved {os.path.normpath(out_path)}")