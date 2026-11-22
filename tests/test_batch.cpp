#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "creditmc/market_risk.hpp"

using namespace creditmc;

namespace {

// Five return series of different lengths.
std::vector<std::vector<double>> make_series() {
    std::mt19937_64 gen(2024);
    std::normal_distribution<double> normal(0.0, 0.01);
    std::vector<std::vector<double>> data;
    for (std::size_t length : {400, 520, 610, 705, 800}) {
        std::vector<double> r(length);
        for (auto& x : r) x = normal(gen);
        data.push_back(std::move(r));
    }
    return data;
}

SeriesList as_spans(const std::vector<std::vector<double>>& data) {
    SeriesList spans;
    for (const auto& r : data) spans.emplace_back(r);
    return spans;
}

bool same(const VarEs& a, const VarEs& b) { return a.var == b.var && a.es == b.es; }

}  // namespace

TEST_CASE("batch results equal the single-series results", "[batch]") {
    const auto data = make_series();
    const auto spans = as_spans(data);

    for (unsigned threads : {1u, 3u, 8u}) {
        CAPTURE(threads);
        const auto hist = historical_var_es_batch(spans, 100, 0.05, threads);
        const auto norm = normal_var_es_batch(spans, 100, 0.05, threads);
        const auto ewma = ewma_var_es_batch(spans, 100, 0.05, 0.94, threads);
        REQUIRE(hist.size() == data.size());
        REQUIRE(norm.size() == data.size());
        REQUIRE(ewma.size() == data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
            CAPTURE(i);
            CHECK(same(hist[i], historical_var_es(data[i], 100, 0.05)));
            CHECK(same(norm[i], normal_var_es(data[i], 100, 0.05)));
            CHECK(same(ewma[i], ewma_var_es(data[i], 100, 0.05, 0.94)));
        }
    }
}

TEST_CASE("batch results do not depend on the thread count", "[batch][threads]") {
    const auto data = make_series();  // must outlive the spans that point into it
    const auto spans = as_spans(data);
    const auto one = historical_var_es_batch(spans, 100, 0.01, 1);
    const auto four = historical_var_es_batch(spans, 100, 0.01, 4);
    const auto automatic = historical_var_es_batch(spans, 100, 0.01, 0);
    REQUIRE(one.size() == four.size());
    for (std::size_t i = 0; i < one.size(); ++i) {
        CHECK(same(one[i], four[i]));
        CHECK(same(one[i], automatic[i]));
    }
}

TEST_CASE("batch handles an empty list and short series", "[batch]") {
    CHECK(historical_var_es_batch({}, 100, 0.05).empty());

    std::vector<double> short_series(50, 0.001);
    std::vector<double> long_series(300, 0.001);
    const SeriesList spans = {short_series, long_series};
    const auto out = historical_var_es_batch(spans, 100, 0.05, 2);
    REQUIRE(out.size() == 2);
    CHECK(out[0].var.empty());       // shorter than the window
    CHECK(out[1].var.size() == 200);
}

TEST_CASE("batch rethrows an error raised in a worker thread", "[batch][validation]") {
    auto data = make_series();
    data[3][10] = std::numeric_limits<double>::quiet_NaN();
    const auto spans = as_spans(data);

    CHECK_THROWS_AS(historical_var_es_batch(spans, 100, 0.05, 4), std::invalid_argument);
    CHECK_THROWS_AS(normal_var_es_batch(spans, 100, 0.05, 1), std::invalid_argument);
    CHECK_THROWS_AS(ewma_var_es_batch(spans, 100, 0.05, 0.94, 8), std::invalid_argument);

    // Bad parameters are reported too.
    const auto clean = make_series();
    CHECK_THROWS_AS(historical_var_es_batch(as_spans(clean), 100, 1.5, 4), std::invalid_argument);
}