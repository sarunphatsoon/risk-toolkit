#include "creditmc/market_risk.hpp"

#include "creditmc/engine.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace creditmc {

namespace {

// Quantile of an already sorted array with linear interpolation, written to match
// numpy.quantile(..., method="linear") bit for bit, including its lerp formula.
double quantile_linear_sorted(const std::vector<double>& sorted, double alpha) {
    const std::size_t n = sorted.size();
    const double virtual_index = static_cast<double>(n - 1) * alpha;
    const std::size_t previous = static_cast<std::size_t>(std::floor(virtual_index));
    if (previous >= n - 1) {
        return sorted[n - 1];
    }
    const double gamma = virtual_index - static_cast<double>(previous);
    const double a = sorted[previous];
    const double b = sorted[previous + 1];
    const double diff = b - a;
    // numpy interpolates from the nearer end for numerical accuracy.
    return gamma >= 0.5 ? b - diff * (1.0 - gamma) : a + diff * gamma;
}

// Replaces one occurrence of `outgoing` by `incoming` in a sorted array, keeping it sorted,
// by shifting only the elements between the two positions (one move, no re-sort).
void replace_in_sorted(std::vector<double>& sorted, double outgoing, double incoming) {
    const auto begin = sorted.begin();
    const std::size_t r = static_cast<std::size_t>(std::lower_bound(begin, sorted.end(), outgoing) - begin);
    const std::size_t p = static_cast<std::size_t>(std::upper_bound(begin, sorted.end(), incoming) - begin);
    if (p > r) {
        // incoming belongs after the removed element: shift (r, p) one step left.
        std::move(begin + static_cast<std::ptrdiff_t>(r) + 1, begin + static_cast<std::ptrdiff_t>(p),
                  begin + static_cast<std::ptrdiff_t>(r));
        sorted[p - 1] = incoming;
    } else {
        // incoming belongs before the removed element: shift [p, r) one step right.
        std::move_backward(begin + static_cast<std::ptrdiff_t>(p), begin + static_cast<std::ptrdiff_t>(r),
                           begin + static_cast<std::ptrdiff_t>(r) + 1);
        sorted[p] = incoming;
    }
}

void validate_common(std::span<const double> returns, double alpha) {
    if (!(alpha > 0.0 && alpha < 1.0)) {
        throw std::invalid_argument("alpha must be in (0, 1)");
    }
    for (double r : returns) {
        if (!std::isfinite(r)) {
            throw std::invalid_argument("returns must be finite");
        }
    }
}

// Runs fn(series[i]) for every series, spread over n_threads threads. Each series is one task,
// handed out dynamically, and each result is written to its own slot, so no locking is needed
// for the results. The first error from any task is rethrown on the calling thread.
template <typename Fn>
std::vector<VarEs> run_batch(const SeriesList& series, unsigned n_threads, Fn fn) {
    std::vector<VarEs> results(series.size());
    if (series.empty()) {
        return results;
    }

    if (n_threads == 0) {
        n_threads = std::max(1u, std::thread::hardware_concurrency());
    }
    n_threads = static_cast<unsigned>(std::min<std::size_t>(n_threads, series.size()));

    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    auto worker = [&]() {
        while (!failed.load()) {
            const std::size_t i = next.fetch_add(1);
            if (i >= series.size()) {
                return;
            }
            try {
                results[i] = fn(series[i]);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
                failed.store(true);
                return;
            }
        }
    };

    std::vector<std::thread> pool;
    for (unsigned t = 1; t < n_threads; ++t) {
        pool.emplace_back(worker);
    }
    worker();  // the calling thread works too
    for (auto& th : pool) {
        th.join();
    }

    if (first_error) {
        std::rethrow_exception(first_error);
    }
    return results;
}

}  // namespace

VarEs historical_var_es(std::span<const double> returns, std::size_t window, double alpha) {
    if (window == 0) {
        throw std::invalid_argument("window must be positive");
    }
    validate_common(returns, alpha);

    VarEs out;
    if (returns.size() <= window) {
        return out;
    }

    const std::size_t n_out = returns.size() - window;
    out.var.reserve(n_out);
    out.es.reserve(n_out);

    // Keep the trailing window sorted. Moving from one day to the next removes one return and
    // adds one, so we update the sorted array in place instead of sorting from scratch.
    std::vector<double> sorted(returns.begin(), returns.begin() + static_cast<std::ptrdiff_t>(window));
    std::sort(sorted.begin(), sorted.end());

    for (std::size_t i = window; i < returns.size(); ++i) {
        // `sorted` now holds days i-window .. i-1.
        const double q = quantile_linear_sorted(sorted, alpha);

        // Tail = every window value <= q (ties included), as in `past[past <= q]`.
        const std::size_t k =
            static_cast<std::size_t>(std::upper_bound(sorted.begin(), sorted.end(), q) - sorted.begin());
        double tail_sum = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            tail_sum += sorted[j];
        }

        out.var.push_back(-q);
        out.es.push_back(-tail_sum / static_cast<double>(k));

        if (i + 1 < returns.size()) {
            replace_in_sorted(sorted, returns[i - window], returns[i]);
        }
    }
    return out;
}

VarEs normal_var_es(std::span<const double> returns, std::size_t window, double alpha) {
    if (window < 2) {
        throw std::invalid_argument("window must be at least 2");
    }
    validate_common(returns, alpha);

    VarEs out;
    if (returns.size() <= window) {
        return out;
    }

    const std::size_t n_out = returns.size() - window;
    out.var.reserve(n_out);
    out.es.reserve(n_out);

    const double z = norm_inv_cdf(alpha);
    const double es_factor = norm_pdf(z) / alpha;
    const double w = static_cast<double>(window);

    for (std::size_t i = window; i < returns.size(); ++i) {
        const auto past = returns.subspan(i - window, window);

        // Two-pass mean and sample variance: more accurate than a running sum of squares.
        double sum = 0.0;
        for (double x : past) {
            sum += x;
        }
        const double mean = sum / w;
        double sum_sq_dev = 0.0;
        for (double x : past) {
            const double d = x - mean;
            sum_sq_dev += d * d;
        }
        const double sd = std::sqrt(sum_sq_dev / (w - 1.0));

        out.var.push_back(-(mean + sd * z));
        out.es.push_back(-mean + sd * es_factor);
    }
    return out;
}

VarEs ewma_var_es(std::span<const double> returns, std::size_t window, double alpha, double lambda) {
    if (window == 0) {
        throw std::invalid_argument("window must be positive");
    }
    if (!(lambda > 0.0 && lambda < 1.0)) {
        throw std::invalid_argument("lambda must be in (0, 1)");
    }
    validate_common(returns, alpha);

    VarEs out;
    if (returns.size() <= window) {
        return out;
    }

    // Variance recursion, started at the first squared return (as pandas ewm(adjust=False) does).
    std::vector<double> v(returns.size());
    v[0] = returns[0] * returns[0];
    for (std::size_t t = 1; t < returns.size(); ++t) {
        v[t] = lambda * v[t - 1] + (1.0 - lambda) * returns[t] * returns[t];
    }

    const std::size_t n_out = returns.size() - window;
    out.var.reserve(n_out);
    out.es.reserve(n_out);

    const double z = norm_inv_cdf(alpha);
    const double es_factor = norm_pdf(z) / alpha;

    for (std::size_t i = window; i < returns.size(); ++i) {
        const double sd = std::sqrt(v[i - 1]);  // the forecast for day i uses variance up to day i-1
        out.var.push_back(-sd * z);
        out.es.push_back(sd * es_factor);
    }
    return out;
}

std::vector<VarEs> historical_var_es_batch(const SeriesList& series, std::size_t window,
                                           double alpha, unsigned n_threads) {
    return run_batch(series, n_threads, [&](std::span<const double> r) {
        return historical_var_es(r, window, alpha);
    });
}

std::vector<VarEs> normal_var_es_batch(const SeriesList& series, std::size_t window,
                                       double alpha, unsigned n_threads) {
    return run_batch(series, n_threads, [&](std::span<const double> r) {
        return normal_var_es(r, window, alpha);
    });
}

std::vector<VarEs> ewma_var_es_batch(const SeriesList& series, std::size_t window, double alpha,
                                     double lambda, unsigned n_threads) {
    return run_batch(series, n_threads, [&](std::span<const double> r) {
        return ewma_var_es(r, window, alpha, lambda);
    });
}

}  // namespace creditmc// namespace creditmc