#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "creditmc/backtest.hpp"

using Catch::Matchers::WithinAbs;
using namespace creditmc;

namespace {

// n observations with the first `x` being hits.
std::vector<int> leading_hits(std::size_t n, std::size_t x) {
    std::vector<int> h(n, 0);
    for (std::size_t i = 0; i < x; ++i) h[i] = 1;
    return h;
}

}  // namespace

TEST_CASE("chi-square survival functions match scipy", "[backtest][chi2]") {
    // Reference values from scipy.stats.chi2.sf.
    CHECK_THAT(chi2_sf_1df(3.841458820694124), WithinAbs(0.04999999999999989, 1e-14));
    CHECK_THAT(chi2_sf_1df(10.0), WithinAbs(0.001565402258002549, 1e-14));
    CHECK_THAT(chi2_sf_1df(0.5), WithinAbs(0.47950012218695337, 1e-14));
    CHECK_THAT(chi2_sf_2df(5.991464547107979), WithinAbs(0.05000000000000007, 1e-14));
    CHECK_THAT(chi2_sf_2df(10.0), WithinAbs(0.006737946999085468, 1e-14));
    CHECK(chi2_sf_1df(0.0) == 1.0);
    CHECK(chi2_sf_1df(-1e-16) == 1.0);
    CHECK(chi2_sf_2df(-0.5) == 1.0);
    CHECK(std::isnan(chi2_sf_1df(std::nan(""))));
}

TEST_CASE("hits uses a strict inequality", "[backtest]") {
    const std::vector<double> r = {-0.03, -0.02, -0.01, 0.01};
    const std::vector<double> v = {0.02, 0.02, 0.02, 0.02};
    const auto h = hits(r, v);
    REQUIRE(h.size() == 4);
    CHECK(h[0] == 1);   // -0.03 < -0.02
    CHECK(h[1] == 0);   // -0.02 is not < -0.02
    CHECK(h[2] == 0);
    CHECK(h[3] == 0);
    CHECK_THROWS_AS(hits(r, std::vector<double>{0.02}), std::invalid_argument);
}

TEST_CASE("kupiec matches reference values", "[backtest][kupiec]") {
    // Reference values computed with the original Python implementation.
    SECTION("10 hits in 250 days at 99%") {
        const auto res = kupiec(leading_hits(250, 10), 0.01);
        CHECK_THAT(res.statistic, WithinAbs(12.955491062356018, 1e-10));
        CHECK_THAT(res.p_value, WithinAbs(0.0003189845082133835, 1e-13));
    }
    SECTION("no hits at all") {
        const auto res = kupiec(leading_hits(250, 0), 0.01);
        CHECK_THAT(res.statistic, WithinAbs(5.025167926750726, 1e-10));
        CHECK_THAT(res.p_value, WithinAbs(0.02498150305344973, 1e-13));
    }
    SECTION("observed rate equals alpha") {
        const auto res = kupiec(leading_hits(100, 5), 0.05);
        CHECK_THAT(res.statistic, WithinAbs(0.0, 1e-12));
        CHECK_THAT(res.p_value, WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("kupiec rejects invalid input", "[backtest][validation]") {
    CHECK_THROWS_AS(kupiec(std::vector<int>{}, 0.05), std::invalid_argument);
    CHECK_THROWS_AS(kupiec(std::vector<int>{0, 1, 2}, 0.05), std::invalid_argument);
    CHECK_THROWS_AS(kupiec(leading_hits(10, 1), 0.0), std::invalid_argument);
    CHECK_THROWS_AS(kupiec(leading_hits(10, 1), 1.0), std::invalid_argument);
}

TEST_CASE("christoffersen independence matches reference values", "[backtest][christoffersen]") {
    SECTION("clustered exceedances") {
        const std::vector<int> seq = {0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
        const auto res = christoffersen_independence(seq);
        CHECK_THAT(res.statistic, WithinAbs(1.3358104147583951, 1e-10));
        CHECK_THAT(res.p_value, WithinAbs(0.2477741635278911, 1e-13));
    }
    SECTION("sequence that starts with a non-hit and ends with a hit") {
        // An additional reference case: the sequence starts in one state and ends in the other.
        const std::vector<int> seq = {0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1};
        const auto res = christoffersen_independence(seq);
        CHECK_THAT(res.statistic, WithinAbs(0.3208070289953824, 1e-10));
        CHECK_THAT(res.p_value, WithinAbs(0.5711230532915781, 1e-13));
    }
    SECTION("no hits gives a statistic of zero") {
        const auto res = christoffersen_independence(std::vector<int>(30, 0));
        CHECK_THAT(res.statistic, WithinAbs(0.0, 1e-12));
        CHECK_THAT(res.p_value, WithinAbs(1.0, 1e-12));
    }
    SECTION("all hits has no defined statistic") {
        const auto res = christoffersen_independence(std::vector<int>(10, 1));
        CHECK(std::isnan(res.statistic));
        CHECK(std::isnan(res.p_value));
    }
    SECTION("too few observations") {
        CHECK_THROWS_AS(christoffersen_independence(std::vector<int>{1}), std::invalid_argument);
    }
}

TEST_CASE("conditional coverage combines both tests", "[backtest][cc]") {
    const std::vector<int> seq = {0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
    const auto res = conditional_coverage(seq, 0.1);
    CHECK_THAT(res.statistic, WithinAbs(7.482353886910335, 1e-10));
    CHECK_THAT(res.p_value, WithinAbs(0.023726162340681856, 1e-13));

    // Consistency with the pieces.
    const auto uc = kupiec(seq, 0.1);
    const auto ind = christoffersen_independence(seq);
    CHECK_THAT(res.statistic, WithinAbs(uc.statistic + ind.statistic, 1e-12));
}

TEST_CASE("traffic light zones", "[backtest][traffic]") {
    CHECK(traffic_light(leading_hits(250, 0)) == "green");
    CHECK(traffic_light(leading_hits(250, 4)) == "green");
    CHECK(traffic_light(leading_hits(250, 5)) == "yellow");
    CHECK(traffic_light(leading_hits(250, 9)) == "yellow");
    CHECK(traffic_light(leading_hits(250, 10)) == "red");

    SECTION("only the last `window` days count") {
        // 10 hits at the start, none in the last 250 days.
        auto h = leading_hits(400, 10);
        CHECK(traffic_light(h) == "green");
        CHECK(traffic_light(h, 400) == "red");
    }
    SECTION("a series shorter than the window uses everything") {
        CHECK(traffic_light(leading_hits(50, 6)) == "yellow");
        CHECK(traffic_light(std::vector<int>{}) == "green");
    }
    CHECK_THROWS_AS(traffic_light(leading_hits(10, 1), 0), std::invalid_argument);
}

TEST_CASE("es_ratio and fz0 match reference values", "[backtest][es]") {
    const std::vector<double> r = {-0.05, 0.01, -0.03, -0.04, 0.02, -0.06, -0.035, 0.0};
    const std::vector<double> var = {0.02, 0.021, 0.022, 0.02, 0.019, 0.02, 0.021, 0.02};
    const std::vector<double> es = {0.028, 0.03, 0.031, 0.027, 0.026, 0.029, 0.03, 0.028};

    // Five exceedances (days 0, 2, 3, 5, 6).
    CHECK_THAT(es_ratio(r, var, es), WithinAbs(1.494113977317537, 1e-14));
    CHECK_THAT(fz0(r, var, es, 0.05), WithinAbs(5.947951097252128, 1e-12));

    SECTION("fewer than five exceedances gives NaN") {
        const std::vector<double> calm = {-0.05, 0.01, -0.03, 0.0, 0.02, 0.0, 0.01, 0.0};
        CHECK(std::isnan(es_ratio(calm, var, es)));
    }
    SECTION("invalid input") {
        CHECK_THROWS_AS(es_ratio(r, std::vector<double>{0.02}, es), std::invalid_argument);
        CHECK_THROWS_AS(fz0(r, var, es, 0.0), std::invalid_argument);
        CHECK_THROWS_AS(fz0(std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, 0.05),
                        std::invalid_argument);
    }
}