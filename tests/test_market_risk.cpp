#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "creditmc/market_risk.hpp"

using Catch::Matchers::WithinAbs;
using namespace creditmc;

TEST_CASE("historical_var_es matches a hand-checked example", "[market][historical]") {
    // Window of 5, alpha 0.25: the quantile position is exactly index 1 of the sorted window.
    const std::vector<double> r = {-0.02, 0.01, -0.01, 0.03, -0.03, 0.00, 0.02, -0.04};
    const auto out = historical_var_es(r, 5, 0.25);

    REQUIRE(out.var.size() == 3);
    REQUIRE(out.es.size() == 3);

    // Day 5 uses days 0-4: sorted {-0.03, -0.02, -0.01, 0.01, 0.03}; q = -0.02; tail {-0.03, -0.02}.
    CHECK_THAT(out.var[0], WithinAbs(0.02, 1e-15));
    CHECK_THAT(out.es[0], WithinAbs(0.025, 1e-15));
    // Day 6 uses days 1-5: sorted {-0.03, -0.01, 0.00, 0.01, 0.03}; q = -0.01; tail {-0.03, -0.01}.
    CHECK_THAT(out.var[1], WithinAbs(0.01, 1e-15));
    CHECK_THAT(out.es[1], WithinAbs(0.02, 1e-15));
    // Day 7 uses days 2-6: sorted {-0.03, -0.01, 0.00, 0.02, 0.03}; q = -0.01; tail {-0.03, -0.01}.
    CHECK_THAT(out.var[2], WithinAbs(0.01, 1e-15));
    CHECK_THAT(out.es[2], WithinAbs(0.02, 1e-15));
}

TEST_CASE("historical_var_es interpolates like numpy", "[market][historical]") {
    SECTION("fraction of exactly one half") {
        // sorted {1, 2, 3, 10}, alpha 0.5: position 1.5 -> halfway between 2 and 3.
        const auto out = historical_var_es(std::vector<double>{1, 2, 3, 10, 99}, 4, 0.5);
        REQUIRE(out.var.size() == 1);
        CHECK_THAT(out.var[0], WithinAbs(-2.5, 1e-15));
        CHECK_THAT(out.es[0], WithinAbs(-1.5, 1e-15));  // tail {1, 2}
    }
    SECTION("fraction below one half") {
        // sorted {0, 10, 20, 30, 40}, alpha 0.1: position 0.4 -> 0 + 0.4 * 10 = 4.
        const auto out = historical_var_es(std::vector<double>{10, 0, 40, 20, 30, 5}, 5, 0.1);
        REQUIRE(out.var.size() == 1);
        CHECK_THAT(out.var[0], WithinAbs(-4.0, 1e-15));
        CHECK_THAT(out.es[0], WithinAbs(0.0, 1e-15));  // tail {0}
    }
}

TEST_CASE("historical_var_es includes tied values in the tail", "[market][historical]") {
    // sorted {-1, -1, -1, 2, 3}, alpha 0.5: quantile is -1, and all three -1 values are in the tail.
    const auto out = historical_var_es(std::vector<double>{-1, -1, -1, 2, 3, 0}, 5, 0.5);
    REQUIRE(out.var.size() == 1);
    CHECK_THAT(out.var[0], WithinAbs(1.0, 1e-15));
    CHECK_THAT(out.es[0], WithinAbs(1.0, 1e-15));
}

TEST_CASE("historical_var_es output has the right length", "[market][historical]") {
    const std::vector<double> r(100, 0.001);
    CHECK(historical_var_es(r, 20, 0.05).var.size() == 80);
    CHECK(historical_var_es(r, 99, 0.05).var.size() == 1);
    CHECK(historical_var_es(r, 100, 0.05).var.empty());
    CHECK(historical_var_es(r, 150, 0.05).var.empty());
}

TEST_CASE("historical_var_es: ES is at least VaR", "[market][historical]") {
    std::mt19937_64 gen(11);
    std::normal_distribution<double> normal(0.0, 0.01);
    std::vector<double> r(1000);
    for (auto& x : r) x = normal(gen);

    const auto out = historical_var_es(r, 250, 0.05);
    REQUIRE(out.var.size() == 750);
    for (std::size_t k = 0; k < out.var.size(); ++k) {
        REQUIRE(out.es[k] >= out.var[k] - 1e-15);
    }
}

TEST_CASE("historical_var_es never looks ahead", "[market][historical]") {
    std::mt19937_64 gen(5);
    std::normal_distribution<double> normal(0.0, 0.01);
    std::vector<double> r(500);
    for (auto& x : r) x = normal(gen);

    const auto base = historical_var_es(r, 100, 0.05);

    // The last return is only ever the "day being forecast", so changing it cannot
    // change any forecast that is returned.
    auto changed = r;
    changed.back() = 0.5;
    const auto after = historical_var_es(changed, 100, 0.05);
    CHECK(base.var == after.var);
    CHECK(base.es == after.es);
}

TEST_CASE("historical_var_es rejects invalid input", "[market][validation]") {
    const std::vector<double> r(50, 0.001);
    CHECK_THROWS_AS(historical_var_es(r, 0, 0.05), std::invalid_argument);
    CHECK_THROWS_AS(historical_var_es(r, 10, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(historical_var_es(r, 10, 1.0), std::invalid_argument);

    auto bad = r;
    bad[7] = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(historical_var_es(bad, 10, 0.05), std::invalid_argument);
    bad[7] = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(historical_var_es(bad, 10, 0.05), std::invalid_argument);
}

TEST_CASE("normal_var_es matches hand-computed reference values", "[market][normal]") {
    // Reference values computed independently with numpy / scipy.
    SECTION("zero-mean window") {
        // Window {0.01, -0.01, 0.03, -0.03}: mean 0, sample sd sqrt(0.002 / 3).
        const auto out = normal_var_es(std::vector<double>{0.01, -0.01, 0.03, -0.03, 0.02}, 4, 0.05);
        REQUIRE(out.var.size() == 1);
        CHECK_THAT(out.var[0], WithinAbs(0.042469938027546145, 1e-14));
        CHECK_THAT(out.es[0], WithinAbs(0.05325901567656669, 1e-14));
    }
    SECTION("non-zero mean") {
        const auto out = normal_var_es(std::vector<double>{0.02, 0.01, 0.03, 0.0, 0.5}, 4, 0.05);
        REQUIRE(out.var.size() == 1);
        CHECK_THAT(out.var[0], WithinAbs(0.006234969013773073, 1e-14));
        CHECK_THAT(out.es[0], WithinAbs(0.011629507838283346, 1e-14));
    }
}

TEST_CASE("normal_var_es: scaling returns scales VaR and ES", "[market][normal]") {
    std::mt19937_64 gen(3);
    std::normal_distribution<double> normal(0.0, 0.01);
    std::vector<double> r(400);
    for (auto& x : r) x = normal(gen);
    auto scaled = r;
    for (auto& x : scaled) x *= 3.0;

    const auto a = normal_var_es(r, 100, 0.05);
    const auto b = normal_var_es(scaled, 100, 0.05);
    REQUIRE(a.var.size() == b.var.size());
    for (std::size_t k = 0; k < a.var.size(); ++k) {
        CHECK_THAT(b.var[k], WithinAbs(3.0 * a.var[k], 1e-12));
        CHECK_THAT(b.es[k], WithinAbs(3.0 * a.es[k], 1e-12));
        CHECK(a.es[k] > a.var[k]);  // ES exceeds VaR for alpha well below 0.5
    }
}

TEST_CASE("normal_var_es rejects invalid input and handles short series", "[market][validation]") {
    const std::vector<double> r(50, 0.001);
    CHECK_THROWS_AS(normal_var_es(r, 1, 0.05), std::invalid_argument);
    CHECK_THROWS_AS(normal_var_es(r, 10, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(normal_var_es(r, 10, 1.0), std::invalid_argument);
    CHECK(normal_var_es(r, 50, 0.05).var.empty());
}

TEST_CASE("ewma_var_es matches hand-computed reference values", "[market][ewma]") {
    // Returns {0.01, -0.02, 0.03}, lambda 0.94: v0 = 1e-4, v1 = 0.94 * 1e-4 + 0.06 * 4e-4 = 1.18e-4.
    // The forecast for day t uses the variance up to day t-1.
    const auto out = ewma_var_es(std::vector<double>{0.01, -0.02, 0.03}, 1, 0.05, 0.94);
    REQUIRE(out.var.size() == 2);
    CHECK_THAT(out.var[0], WithinAbs(0.01644853626951473, 1e-15));   // sd = sqrt(v0) = 0.01
    CHECK_THAT(out.es[0], WithinAbs(0.020627128075074254, 1e-15));
    CHECK_THAT(out.var[1], WithinAbs(0.017867683889728377, 1e-15));  // sd = sqrt(v1)
    CHECK_THAT(out.es[1], WithinAbs(0.022406796444340488, 1e-15));
}

TEST_CASE("ewma_var_es never looks ahead", "[market][ewma]") {
    std::mt19937_64 gen(9);
    std::normal_distribution<double> normal(0.0, 0.01);
    std::vector<double> r(300);
    for (auto& x : r) x = normal(gen);

    const auto base = ewma_var_es(r, 50, 0.05, 0.94);
    auto changed = r;
    changed.back() = 0.5;  // only affects the (not returned) forecast for the day after
    const auto after = ewma_var_es(changed, 50, 0.05, 0.94);
    CHECK(base.var == after.var);
    CHECK(base.es == after.es);
    REQUIRE(base.var.size() == 250);
}

TEST_CASE("ewma_var_es rejects invalid input", "[market][validation]") {
    const std::vector<double> r(50, 0.001);
    CHECK_THROWS_AS(ewma_var_es(r, 0, 0.05, 0.94), std::invalid_argument);
    CHECK_THROWS_AS(ewma_var_es(r, 10, 0.05, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(ewma_var_es(r, 10, 0.05, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(ewma_var_es(r, 10, 1.0, 0.94), std::invalid_argument);
    CHECK(ewma_var_es(r, 50, 0.05, 0.94).var.empty());
}
