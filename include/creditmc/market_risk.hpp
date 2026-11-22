#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace creditmc {

// Rolling VaR / Expected Shortfall forecasts. Both are positive loss numbers.
// Element k of each vector is the forecast for day (window + k): it uses only
// returns[k, k + window), never the day being forecast.
struct VarEs {
    std::vector<double> var;
    std::vector<double> es;
};

// Rolling historical-simulation VaR and ES.
//   VaR = -(alpha-quantile of the window), using numpy's default linear interpolation.
//   ES  = -(mean of the window values that are <= that quantile).
// Returns empty vectors if returns.size() <= window.
// Throws std::invalid_argument if window == 0, alpha is outside (0, 1), or any return
// is not finite.
VarEs historical_var_es(std::span<const double> returns, std::size_t window, double alpha);

// Rolling normal (variance-covariance) VaR and ES from the trailing window's sample mean
// and sample standard deviation (n - 1 divisor):
//   VaR = -(mu + sd * z),   ES = -mu + sd * pdf(z) / alpha,   z = Phi^-1(alpha).
// Throws std::invalid_argument if window < 2, alpha is outside (0, 1), or a return is
// not finite. Returns empty vectors if returns.size() <= window.
VarEs normal_var_es(std::span<const double> returns, std::size_t window, double alpha);

// EWMA (RiskMetrics-style) normal VaR and ES with zero mean:
//   v_0 = r_0^2,   v_t = lambda * v_{t-1} + (1 - lambda) * r_t^2,
//   VaR = -sd * z,   ES = sd * pdf(z) / alpha,   sd = sqrt(v_{t-1}) for the forecast of day t.
// Output starts at day `window` (the earlier days are a burn-in for the variance estimate).
// Throws std::invalid_argument if window == 0, alpha or lambda is outside (0, 1), or a return
// is not finite. Returns empty vectors if returns.size() <= window.
VarEs ewma_var_es(std::span<const double> returns, std::size_t window, double alpha, double lambda);

// ---------------------------------------------------------------------------
// Batch versions: many return series at once, processed in parallel across series.
// Series may have different lengths. Results are identical for any thread count, and
// result[i] is the same as the single-series function applied to series[i].
// n_threads = 0 means "use all hardware threads". An error in any series (for example a
// non-finite return) is rethrown to the caller. An empty list gives an empty result.
// A single series is still processed by one thread.
// ---------------------------------------------------------------------------
using SeriesList = std::vector<std::span<const double>>;

std::vector<VarEs> historical_var_es_batch(const SeriesList& series, std::size_t window,
                                           double alpha, unsigned n_threads = 0);

std::vector<VarEs> normal_var_es_batch(const SeriesList& series, std::size_t window,
                                       double alpha, unsigned n_threads = 0);

std::vector<VarEs> ewma_var_es_batch(const SeriesList& series, std::size_t window, double alpha,
                                     double lambda, unsigned n_threads = 0);

}  // namespace creditmc