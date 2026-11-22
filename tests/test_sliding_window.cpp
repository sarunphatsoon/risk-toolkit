#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "creditmc/market_risk.hpp"

using namespace creditmc;

namespace {

// Independent brute-force version: sort every window from scratch. It uses the same
// linear-interpolation quantile and the same "mean of values <= q" tail as the library.
VarEs brute_force(const std::vector<double>& r, std::size_t window, double alpha) {
    VarEs out;
    for (std::size_t i = window; i < r.size(); ++i) {
        std::vector<double> w(r.begin() + static_cast<std::ptrdiff_t>(i - window),
                              r.begin() + static_cast<std::ptrdiff_t>(i));
        std::sort(w.begin(), w.end());
        const double pos = static_cast<double>(window - 1) * alpha;
        const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
        const double gamma = pos - static_cast<double>(lo);
        const double a = w[lo];
        const double b = w[std::min(lo + 1, window - 1)];
        const double q = gamma >= 0.5 ? b - (b - a) * (1.0 - gamma) : a + (b - a) * gamma;
        double sum = 0.0;
        std::size_t k = 0;
        for (double x : w) {
            if (x <= q) {
                sum += x;
                ++k;
            }
        }
        out.var.push_back(-q);
        out.es.push_back(-sum / static_cast<double>(k));
    }
    return out;
}

}  // namespace

TEST_CASE("sliding-window historical VaR/ES equals a brute-force re-sort", "[historical][sliding]") {
    std::mt19937_64 gen(77);

    SECTION("continuous returns") {
        std::normal_distribution<double> normal(0.0, 0.01);
        std::vector<double> r(900);
        for (auto& x : r) x = normal(gen);
        for (std::size_t window : {1u, 2u, 7u, 60u, 250u}) {
            for (double alpha : {0.01, 0.05, 0.5, 0.95}) {
                CAPTURE(window, alpha);
                const auto got = historical_var_es(r, window, alpha);
                const auto want = brute_force(r, window, alpha);
                CHECK(got.var == want.var);
                CHECK(got.es == want.es);
            }
        }
    }

    SECTION("heavily tied returns (only 9 distinct values)") {
        std::uniform_int_distribution<int> pick(-4, 4);
        std::vector<double> r(900);
        for (auto& x : r) x = pick(gen) / 100.0;
        for (std::size_t window : {5u, 20u, 100u}) {
            for (double alpha : {0.05, 0.25, 0.5}) {
                CAPTURE(window, alpha);
                const auto got = historical_var_es(r, window, alpha);
                const auto want = brute_force(r, window, alpha);
                CHECK(got.var == want.var);
                CHECK(got.es == want.es);
            }
        }
    }

    SECTION("a monotonic series and a constant series") {
        std::vector<double> up(300), flat(300, 0.01);
        for (std::size_t i = 0; i < up.size(); ++i) up[i] = static_cast<double>(i) / 1000.0;
        CHECK(historical_var_es(up, 50, 0.05).var == brute_force(up, 50, 0.05).var);
        CHECK(historical_var_es(up, 50, 0.05).es == brute_force(up, 50, 0.05).es);
        CHECK(historical_var_es(flat, 50, 0.05).var == brute_force(flat, 50, 0.05).var);
        CHECK(historical_var_es(flat, 50, 0.05).es == brute_force(flat, 50, 0.05).es);
    }
}