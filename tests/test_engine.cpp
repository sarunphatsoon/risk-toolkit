#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "creditmc/engine.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace creditmc;

namespace {

// A small, valid portfolio used by several tests.
Portfolio make_portfolio(std::size_t n, double pd = 0.05, double ead = 1.0,
                         double lgd = 0.5, double rho = 0.2) {
    return Portfolio{std::vector<double>(n, pd), std::vector<double>(n, ead),
                     std::vector<double>(n, lgd), rho};
}

}  // namespace

TEST_CASE("norm_cdf matches known values", "[normal]") {
    CHECK_THAT(norm_cdf(0.0), WithinAbs(0.5, 1e-12));
    CHECK_THAT(norm_cdf(1.96), WithinAbs(0.9750021048517795, 1e-12));
    CHECK_THAT(norm_cdf(-1.0), WithinAbs(0.15865525393145707, 1e-12));
}

TEST_CASE("norm_inv_cdf matches known values", "[normal]") {
    CHECK_THAT(norm_inv_cdf(0.5), WithinAbs(0.0, 1e-9));
    CHECK_THAT(norm_inv_cdf(0.975), WithinAbs(1.959963984540054, 1e-8));
    CHECK_THAT(norm_inv_cdf(0.001), WithinAbs(-3.090232306167813, 1e-8));
    CHECK_THAT(norm_inv_cdf(0.999), WithinAbs(3.090232306167813, 1e-8));
}

TEST_CASE("norm_cdf(norm_inv_cdf(p)) round-trips", "[normal]") {
    for (double p : {1e-6, 1e-3, 0.02, 0.1, 0.5, 0.9, 0.999, 1.0 - 1e-6}) {
        CAPTURE(p);
        CHECK_THAT(norm_cdf(norm_inv_cdf(p)), WithinRel(p, 1e-6));
    }
}

TEST_CASE("norm_inv_cdf is antisymmetric around 0.5", "[normal]") {
    for (double p : {0.001, 0.01, 0.1, 0.3}) {
        CAPTURE(p);
        CHECK_THAT(norm_inv_cdf(p), WithinAbs(-norm_inv_cdf(1.0 - p), 1e-9));
    }
}

TEST_CASE("norm_inv_cdf rejects probabilities outside (0, 1)", "[normal]") {
    CHECK_THROWS_AS(norm_inv_cdf(0.0), std::invalid_argument);
    CHECK_THROWS_AS(norm_inv_cdf(1.0), std::invalid_argument);
    CHECK_THROWS_AS(norm_inv_cdf(-0.1), std::invalid_argument);
    CHECK_THROWS_AS(norm_inv_cdf(std::numeric_limits<double>::quiet_NaN()),
                    std::invalid_argument);
}

TEST_CASE("IRB formula matches independently computed values", "[irb]") {
    // Reference values computed with SciPy.
    CHECK_THAT(irb_asymptotic_loss(0.02, 0.45, 0.15, 0.999),
               WithinAbs(0.0793480226157891, 1e-8));
    CHECK_THAT(irb_asymptotic_loss(0.01, 0.40, 0.12, 0.99),
               WithinAbs(0.021010636851525842, 1e-8));
}

TEST_CASE("IRB loss increases with correlation and confidence", "[irb]") {
    CHECK(irb_asymptotic_loss(0.02, 0.45, 0.10, 0.999) <
          irb_asymptotic_loss(0.02, 0.45, 0.20, 0.999));
    CHECK(irb_asymptotic_loss(0.02, 0.45, 0.15, 0.99) <
          irb_asymptotic_loss(0.02, 0.45, 0.15, 0.999));
}

TEST_CASE("simulate_losses rejects invalid input", "[validation]") {
    SECTION("empty portfolio") {
        Portfolio p{{}, {}, {}, 0.2};
        CHECK_THROWS_AS(simulate_losses(p, 10, 1), std::invalid_argument);
    }
    SECTION("mismatched lengths") {
        Portfolio p = make_portfolio(5);
        p.ead.pop_back();
        CHECK_THROWS_AS(simulate_losses(p, 10, 1), std::invalid_argument);
    }
    SECTION("rho outside (0, 1)") {
        CHECK_THROWS_AS(simulate_losses(make_portfolio(5, 0.05, 1.0, 0.5, 0.0), 10, 1),
                        std::invalid_argument);
        CHECK_THROWS_AS(simulate_losses(make_portfolio(5, 0.05, 1.0, 0.5, 1.0), 10, 1),
                        std::invalid_argument);
    }
    SECTION("zero scenarios") {
        CHECK_THROWS_AS(simulate_losses(make_portfolio(5), 0, 1), std::invalid_argument);
    }
    SECTION("pd outside (0, 1)") {
        CHECK_THROWS_AS(simulate_losses(make_portfolio(5, 0.0), 10, 1), std::invalid_argument);
        CHECK_THROWS_AS(simulate_losses(make_portfolio(5, 1.0), 10, 1), std::invalid_argument);
    }
    SECTION("lgd outside [0, 1]") {
        CHECK_THROWS_AS(simulate_losses(make_portfolio(5, 0.05, 1.0, 1.5), 10, 1),
                        std::invalid_argument);
    }
    SECTION("negative exposure") {
        CHECK_THROWS_AS(simulate_losses(make_portfolio(5, 0.05, -1.0), 10, 1),
                        std::invalid_argument);
    }
}

TEST_CASE("simulate_losses output has the right shape and bounds", "[simulate]") {
    const auto p = make_portfolio(50);
    const auto losses = simulate_losses(p, 3000, 7);
    REQUIRE(losses.size() == 3000);
    const double max_loss = std::accumulate(p.ead.begin(), p.ead.end(), 0.0) * 0.5;
    for (double x : losses) {
        REQUIRE(x >= 0.0);
        REQUIRE(x <= max_loss + 1e-12);
    }
}

TEST_CASE("results do not depend on the thread count", "[simulate][threads]") {
    const auto p = make_portfolio(50);
    const std::size_t n = 5000;  // spans several 1024-scenario blocks
    const auto one = simulate_losses(p, n, 123, 1);
    const auto four = simulate_losses(p, n, 123, 4);
    const auto automatic = simulate_losses(p, n, 123, 0);
    CHECK(one == four);
    CHECK(one == automatic);
}

TEST_CASE("same seed repeats and different seeds differ", "[simulate]") {
    const auto p = make_portfolio(50);
    CHECK(simulate_losses(p, 2000, 5) == simulate_losses(p, 2000, 5));
    CHECK(simulate_losses(p, 2000, 5) != simulate_losses(p, 2000, 6));
}

TEST_CASE("simulated mean loss matches analytical expected loss", "[simulate][statistics]") {
    const std::size_t n_obligors = 100;
    const auto p = make_portfolio(n_obligors, 0.05, 1.0, 0.5, 0.2);
    const double expected_loss = n_obligors * 0.05 * 1.0 * 0.5;  // 2.5

    const auto losses = simulate_losses(p, 200000, 42);
    const double mean = std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();

    // Standard error is about 0.005; allow 0.05 so the test is robust.
    CHECK_THAT(mean, WithinAbs(expected_loss, 0.05));
}

TEST_CASE("zero LGD means zero losses", "[simulate]") {
    const auto losses = simulate_losses(make_portfolio(20, 0.05, 1.0, 0.0), 1000, 3);
    for (double x : losses) {
        REQUIRE(x == 0.0);
    }
}

TEST_CASE("99% VaR of a large granular portfolio is close to the IRB value",
          "[simulate][statistics]") {
    // With many small, equal loans the asymptotic (IRB) formula should be accurate.
    // This test depends on the correlation structure, which a mean-loss test cannot see.
    const std::size_t n_obligors = 2000;
    const double total_ead = 1000.0, pd = 0.02, lgd = 0.45, rho = 0.15, confidence = 0.99;
    const Portfolio p{std::vector<double>(n_obligors, pd),
                      std::vector<double>(n_obligors, total_ead / n_obligors),
                      std::vector<double>(n_obligors, lgd), rho};

    auto losses = simulate_losses(p, 100000, 2024);
    std::sort(losses.begin(), losses.end());
    const double var = losses[static_cast<std::size_t>(confidence * losses.size())];
    const double irb = total_ead * irb_asymptotic_loss(pd, lgd, rho, confidence);

    CHECK_THAT(var, WithinRel(irb, 0.04));
}

TEST_CASE("norm_inv_cdf agrees with SciPy to near machine precision", "[normal]") {
    // Reference values from scipy.stats.norm.ppf.
    CHECK_THAT(norm_inv_cdf(1e-9), WithinAbs(-5.9978070150076865, 1e-13));
    CHECK_THAT(norm_inv_cdf(1e-6), WithinAbs(-4.753424308822899, 1e-13));
    CHECK_THAT(norm_inv_cdf(0.001), WithinAbs(-3.090232306167813, 1e-13));
    CHECK_THAT(norm_inv_cdf(0.05), WithinAbs(-1.6448536269514729, 1e-13));
    CHECK_THAT(norm_inv_cdf(0.3), WithinAbs(-0.5244005127080409, 1e-13));
    CHECK_THAT(norm_inv_cdf(0.975), WithinAbs(1.959963984540054, 1e-13));
    CHECK_THAT(norm_inv_cdf(0.999999), WithinAbs(4.753424308817087, 1e-12));
}

TEST_CASE("norm_pdf matches known values", "[normal]") {
    CHECK_THAT(norm_pdf(0.0), WithinAbs(0.3989422804014327, 1e-15));
    CHECK_THAT(norm_pdf(1.0), WithinAbs(0.24197072451914337, 1e-15));
    CHECK_THAT(norm_pdf(-2.5), WithinAbs(0.017528300493568537, 1e-15));
}