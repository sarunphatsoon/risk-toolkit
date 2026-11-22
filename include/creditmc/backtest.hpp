#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace creditmc {

// Result of a statistical test: the likelihood-ratio statistic and its p-value.
struct TestResult {
    double statistic;
    double p_value;
};

// Chi-square survival functions P(X > x), in closed form for 1 and 2 degrees of freedom.
// They return 1 for x <= 0 (as scipy does) and NaN for NaN.
double chi2_sf_1df(double x);
double chi2_sf_2df(double x);

// Exceedance indicators: hit[i] = 1 if returns[i] < -var[i], else 0 (strict inequality).
// returns and var must have the same length (already aligned, one entry per forecast day).
std::vector<int> hits(std::span<const double> returns, std::span<const double> var);

// Kupiec proportion-of-failures test of unconditional coverage (1 degree of freedom).
// Throws std::invalid_argument if hits is empty or not 0/1, or alpha is outside (0, 1).
TestResult kupiec(std::span<const int> hits, double alpha);

// Christoffersen test of independence of exceedances (1 degree of freedom), using the
// n - 1 day-to-day transitions. Returns {NaN, NaN} when there is no "previous day without
// a hit" at all (n00 + n01 == 0). Throws if fewer than 2 observations or not 0/1.
TestResult christoffersen_independence(std::span<const int> hits);

// Conditional coverage: Kupiec + Christoffersen statistics, compared with a chi-square
// with 2 degrees of freedom.
TestResult conditional_coverage(std::span<const int> hits, double alpha);

// Mean of (loss / ES) over the days on which the VaR was exceeded. Returns NaN if there are
// fewer than 5 exceedances. The t-test of this ratio against 1 is not included.
double es_ratio(std::span<const double> returns, std::span<const double> var,
                std::span<const double> es);

// Basel traffic-light zone from the number of exceedances in the last `window` days:
// "green" for up to 4, "yellow" for 5 to 9, "red" for 10 or more. These zones are defined
// for 99% VaR over 250 days; they are not meaningful for other confidence levels.
std::string traffic_light(std::span<const int> hits, std::size_t window = 250);

// Mean Fissler-Ziegel (FZ0) joint score of VaR and ES forecasts (lower is better). `var` and
// `es` are positive losses. es must be positive, otherwise the result is NaN or infinite.
double fz0(std::span<const double> returns, std::span<const double> var,
           std::span<const double> es, double alpha);

}  // namespace creditmc