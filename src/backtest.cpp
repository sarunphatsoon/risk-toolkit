#include "creditmc/backtest.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace creditmc {

namespace {

// x * log(y), defined as 0 when x == 0 (like scipy.special.xlogy).
double xlogy(double x, double y) {
    if (x == 0.0 && !std::isnan(y)) {
        return 0.0;
    }
    return x * std::log(y);
}

// Bernoulli log-likelihood of n_fail failures and n_ok successes with failure probability p.
double log_likelihood(double n_fail, double n_ok, double p) {
    return xlogy(n_ok, 1.0 - p) + xlogy(n_fail, p);
}

void validate_hits(std::span<const int> h, std::size_t min_size) {
    if (h.size() < min_size) {
        throw std::invalid_argument("not enough observations");
    }
    for (int x : h) {
        if (x != 0 && x != 1) {
            throw std::invalid_argument("hits must be 0 or 1");
        }
    }
}

void validate_alpha(double alpha) {
    if (!(alpha > 0.0 && alpha < 1.0)) {
        throw std::invalid_argument("alpha must be in (0, 1)");
    }
}

}  // namespace

double chi2_sf_1df(double x) {
    if (x <= 0.0) {
        return 1.0;
    }
    return std::erfc(std::sqrt(0.5 * x));
}

double chi2_sf_2df(double x) {
    if (x <= 0.0) {
        return 1.0;
    }
    return std::exp(-0.5 * x);
}

std::vector<int> hits(std::span<const double> returns, std::span<const double> var) {
    if (returns.size() != var.size()) {
        throw std::invalid_argument("returns and var must have the same length");
    }
    std::vector<int> out(returns.size());
    for (std::size_t i = 0; i < returns.size(); ++i) {
        out[i] = returns[i] < -var[i] ? 1 : 0;
    }
    return out;
}

TestResult kupiec(std::span<const int> h, double alpha) {
    validate_hits(h, 1);
    validate_alpha(alpha);

    const double n = static_cast<double>(h.size());
    double x = 0.0;
    for (int v : h) {
        x += v;
    }
    const double observed_rate = x / n;
    const double lr = -2.0 * (log_likelihood(x, n - x, alpha) - log_likelihood(x, n - x, observed_rate));
    return {lr, chi2_sf_1df(lr)};
}

TestResult christoffersen_independence(std::span<const int> h) {
    validate_hits(h, 2);

    double n00 = 0, n01 = 0, n10 = 0, n11 = 0;
    for (std::size_t t = 1; t < h.size(); ++t) {
        const int prev = h[t - 1];
        const int curr = h[t];
        if (prev == 0 && curr == 0) ++n00;
        if (prev == 0 && curr == 1) ++n01;
        if (prev == 1 && curr == 0) ++n10;
        if (prev == 1 && curr == 1) ++n11;
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (n00 + n01 == 0.0) {
        return {nan, nan};
    }
    const double pi01 = n01 / (n00 + n01);
    const double pi11 = (n10 + n11) > 0.0 ? n11 / (n10 + n11) : 0.0;
    const double pi = (n01 + n11) / (n00 + n01 + n10 + n11);

    const double ll0 = log_likelihood(n01 + n11, n00 + n10, pi);
    const double ll1 = log_likelihood(n01, n00, pi01) + log_likelihood(n11, n10, pi11);
    const double lr = -2.0 * (ll0 - ll1);
    return {lr, chi2_sf_1df(lr)};
}

TestResult conditional_coverage(std::span<const int> h, double alpha) {
    const TestResult uc = kupiec(h, alpha);
    const TestResult ind = christoffersen_independence(h);
    const double lr = uc.statistic + ind.statistic;
    return {lr, chi2_sf_2df(lr)};
}

double es_ratio(std::span<const double> returns, std::span<const double> var,
                std::span<const double> es) {
    if (returns.size() != var.size() || returns.size() != es.size()) {
        throw std::invalid_argument("returns, var and es must have the same length");
    }
    std::size_t n_hits = 0;
    double sum = 0.0;
    for (std::size_t i = 0; i < returns.size(); ++i) {
        if (returns[i] < -var[i]) {
            ++n_hits;
            sum += (-returns[i]) / es[i];
        }
    }
    if (n_hits < 5) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum / static_cast<double>(n_hits);
}

std::string traffic_light(std::span<const int> h, std::size_t window) {
    if (window == 0) {
        throw std::invalid_argument("window must be positive");
    }
    validate_hits(h, 0);
    const std::size_t count = h.size() < window ? h.size() : window;
    int exceptions = 0;
    for (std::size_t i = h.size() - count; i < h.size(); ++i) {
        exceptions += h[i];
    }
    if (exceptions <= 4) return "green";
    if (exceptions <= 9) return "yellow";
    return "red";
}

double fz0(std::span<const double> returns, std::span<const double> var,
           std::span<const double> es, double alpha) {
    if (returns.size() != var.size() || returns.size() != es.size()) {
        throw std::invalid_argument("returns, var and es must have the same length");
    }
    if (returns.empty()) {
        throw std::invalid_argument("not enough observations");
    }
    validate_alpha(alpha);

    double total = 0.0;
    for (std::size_t i = 0; i < returns.size(); ++i) {
        const double r = returns[i];
        const double v = -var[i];
        const double e = -es[i];
        const double exceed_term = (r <= v) ? -(1.0 / (alpha * e)) * (v - r) : 0.0;
        total += exceed_term + v / e + std::log(-e) - 1.0;
    }
    return total / static_cast<double>(returns.size());
}

}  // namespace creditmc