import os 
import time

import matplotlib.pyplot as plt
import numpy as np

import creditmc_core as cpp
from reference import simulate_losses_numpy

N_OBLIGORS = 1000
N_SCENARIOS = 200_000
N_REPEATS = 5
PD, EAD, LGD, RHO = 0.02, 1.0, 0.45, 0.15

pd = np.full(N_OBLIGORS, PD)
ead = np.full(N_OBLIGORS, EAD)
lgd = np.full(N_OBLIGORS, LGD)

max_cpu = os.cpu_count() or 1
thread_counts = sorted({t for t in (1, 2, 4, 8, 16) if t <= max_cpu} | {max_cpu})
print(f"Logical CPUs: {max_cpu}; testing thread counts {thread_counts}\n")

cpp.simulate(pd, ead, lgd, RHO, 20_000, seed = 0)

#Numpy baseline
numpy_times = []
for _ in range(3):
    t0 = time.perf_counter()
    simulate_losses_numpy(pd, ead, lgd, RHO, N_SCENARIOS, seed = 1)
    numpy_times.append(time.perf_counter() - t0)
numpy_time = np.median(numpy_times)
print(f"NumPy (1 core): median {numpy_time:.3f}s\n")

median_time = []
reference = None
for n_threads in thread_counts:
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        losses = cpp.simulate(pd, ead, lgd, RHO, N_SCENARIOS, seed = 1, n_threads = n_threads)
        times.append(time.perf_counter() - t0)
    median_time.append(np.median(times))

    if reference is None:
        reference = losses
    identical = np.array_equal(reference, losses)
    print(f"{n_threads:3d} threads: median {median_time[-1]:6.3f}s   "
          f"(min {min(times):.3f}s, max {max(times):.3f}s)    identical results: {identical}")

counts = np.array(thread_counts, dtype = float)
median_time = np.array(median_time)
speedup = median_time[0] / median_time
efficiency = speedup / counts * 100

print(f"\n{'threads':>8}{'time (s)':10}{'speedup':>9}{'efficiency':>12}")
for n, t, s, e in zip(thread_counts, median_time, speedup, efficiency):
    print(f"{n:>8}{t:>10.3f}{s:>8.2f}x{e:>11.0f}%")

print(f"\nC++ at 1 thread vs Numpy: {numpy_time / median_time[0]:.1f}x faster")
print(f"C++ at {thread_counts[-1]} threads vs Numpy: {numpy_time / median_time[-1]:.1f}x faster")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))


ax1.plot(counts, counts, linestyle = '--', color = "grey", label = "Ideal (linear)")
ax1.plot(counts, speedup, marker = 'o', label = "Measured")
ax1.set_xlabel("Threads")
ax1.set_ylabel("Speedup vs 1 thread")
ax1.set_title("Thread scaling")
ax1.set_xticks(counts)
ax1.grid(alpha = 0.3)
ax1.legend()

ax2.plot(counts, efficiency, marker = "o", color = "tab:orange")
ax2.axhline(100, linestyle = "--", color = "grey")
ax2.set_xlabel("Threads")
ax2.set_ylabel("Parallel efficiency (%)")
ax2.set_title("Efficiency (speedup / threads)")
ax2.set_xticks(counts)
ax2.set_ylim(0, 110)
ax2.grid(alpha = 0.3)

fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
os.makedirs(out_dir, exist_ok = True)
out_path = os.path.join(out_dir, "threads.png")
fig.savefig(out_path, dpi = 150)
print(f"\nSaved {os.path.normpath(out_path)}")